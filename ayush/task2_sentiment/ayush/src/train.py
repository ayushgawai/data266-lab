"""Train TextCNN, BiLSTM, and BiLSTM+attention on Yelp polarity."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common.runtime import load_config, open_run_log, peak_mem_gb, resolve_device, set_seeds, write_manifest
from task2_sentiment.ayush.src.data import (
    build_vocab,
    clean_tokens,
    encode_rows,
    make_stratified_indices,
    slice_flags,
)
from task2_sentiment.ayush.src.models import BiLSTMAttn, BiLSTMPool, TextCNN

STOP = set(ENGLISH_STOP_WORDS)


def load_yelp():
    from datasets import load_dataset

    # Canonical "yelp_polarity" has no namespace and breaks current huggingface_hub.
    ds = load_dataset("fancyzhx/yelp_polarity")
    if len(ds["train"]) != 560000 or len(ds["test"]) != 38000:
        raise SystemExit(f"unexpected yelp sizes train={len(ds['train'])} test={len(ds['test'])}")
    def pack(split):
        labels = np.asarray(ds[split]["label"], dtype=np.int64)
        texts = list(ds[split]["text"])
        return texts, labels
    return pack("train"), pack("test")


def lemmatizer():
    import nltk
    from nltk.stem import WordNetLemmatizer

    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    lem = WordNetLemmatizer()
    return lem.lemmatize


def tokenize_rows(texts, indices, use_stop, use_lemma, lemma_fn, drop_dups: bool = True):
    """Train and val drop empty and duplicate cleaned text. Test keeps every official row."""
    stop = STOP if use_stop else None
    lem = lemma_fn if use_lemma else None
    rows, keep = [], []
    n_empty = 0
    seen = set()
    n_dup = 0
    for i in indices:
        toks = clean_tokens(texts[int(i)], stop=stop, lemmatize=lem)
        key = " ".join(toks)
        if not toks:
            n_empty += 1
            if drop_dups:
                continue
            toks = ["unk"]
        if drop_dups and key in seen:
            n_dup += 1
            continue
        seen.add(key)
        rows.append(toks)
        keep.append(int(i))
    return rows, np.asarray(keep, dtype=np.int64), n_empty, n_dup


def make_loader(x, y, lengths, batch, shuffle):
    ds = TensorDataset(
        torch.from_numpy(x.astype(np.int64)),
        torch.from_numpy(y.astype(np.int64)),
        torch.from_numpy(lengths.astype(np.int64)),
    )
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def run_epoch(model, loader, opt, device, train: bool):
    model.train(train)
    total, n = 0.0, 0
    probs, gold = [], []
    for x, y, lengths in loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        if train:
            opt.zero_grad(set_to_none=True)
        logits = model(x, lengths)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        if train:
            loss.backward()
            opt.step()
        total += loss.item() * y.size(0)
        n += y.size(0)
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        gold.append(y.detach().cpu().numpy())
    return total / max(1, n), np.concatenate(probs), np.concatenate(gold)


def ece_score(y, p, n_bins=15):
    conf = np.maximum(p, 1.0 - p)
    pred = (p >= 0.5).astype(np.int64)
    correct = (pred == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    for i in range(n_bins):
        hi = bins[i + 1]
        m = (conf >= bins[i]) & (conf <= hi if i == n_bins - 1 else conf < hi)
        if not np.any(m):
            continue
        out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def metric_row(name, y, p, flags, n_params, seconds, examples_per_sec, mem):
    pred = (p >= 0.5).astype(np.int64)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average=None, labels=[0, 1], zero_division=0)
    mac = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    mic = precision_recall_fscore_support(y, pred, average="micro", zero_division=0)
    wgt = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    rng = np.random.default_rng(42)
    accs, f1s, mccs = [], [], []
    for _ in range(1000):
        ix = rng.integers(0, len(y), len(y))
        accs.append(accuracy_score(y[ix], pred[ix]))
        f1s.append(f1_score(y[ix], pred[ix], average="macro", zero_division=0))
        mccs.append(matthews_corrcoef(y[ix], pred[ix]))
    lo, hi = np.percentile(accs, [2.5, 97.5])
    f_lo, f_hi = np.percentile(f1s, [2.5, 97.5])
    m_lo, m_hi = np.percentile(mccs, [2.5, 97.5])
    row = {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "precision_macro": mac[0], "recall_macro": mac[1], "f1_macro": mac[2],
        "precision_micro": mic[0], "recall_micro": mic[1], "f1_micro": mic[2],
        "precision_weighted": wgt[0], "recall_weighted": wgt[1], "f1_weighted": wgt[2],
        "precision_neg": pr[0], "recall_neg": rc[0], "f1_neg": f1[0],
        "precision_pos": pr[1], "recall_pos": rc[1], "f1_pos": f1[1],
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "mcc": matthews_corrcoef(y, pred),
        "brier": brier_score_loss(y, p),
        "ece": ece_score(y, p),
        "acc_ci95_lo": lo, "acc_ci95_hi": hi,
        "f1_macro_ci95_lo": f_lo, "f1_macro_ci95_hi": f_hi,
        "mcc_ci95_lo": m_lo, "mcc_ci95_hi": m_hi,
        "param_count": n_params,
        "train_time_sec": seconds,
        "examples_per_sec": examples_per_sec,
        "peak_mem_gb": mem,
    }
    for key, mask in flags.items():
        if mask.sum() == 0:
            row[f"f1_{key}"] = ""
            row[f"n_{key}"] = 0
            continue
        row[f"f1_{key}"] = f1_score(y[mask], pred[mask], average="macro", zero_division=0)
        row[f"n_{key}"] = int(mask.sum())
        row[f"err_{key}"] = float((pred[mask] != y[mask]).mean())
    return row


def mcnemar(y, pred_a, pred_b):
    a_ok = pred_a == y
    b_ok = pred_b == y
    b01 = int((~a_ok & b_ok).sum())
    b10 = int((a_ok & ~b_ok).sum())
    stat = (abs(b01 - b10) - 1) ** 2 / max(1, b01 + b10)
    # chi-square survival with 1 df. series is enough, no scipy required here.
    # P(chi^2_1 > x) = erfc(sqrt(x/2))
    import math
    p = math.erfc(math.sqrt(stat / 2.0))
    return {"n10": b10, "n01": b01, "stat": stat, "p": p}


def build_model(name, vocab):
    if name == "textcnn":
        return TextCNN(vocab)
    if name == "bilstm":
        return BiLSTMPool(vocab)
    if name == "bilstm_attn":
        return BiLSTMAttn(vocab)
    raise ValueError(name)


def fit(name, vocab, train_loader, val_loader, device, epochs, lr, patience, log):
    model = build_model(name, vocab).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, stall = float("inf"), None, 0
    hist = []
    t0 = time.time()
    seen = 0
    for epoch in range(1, epochs + 1):
        tr, _, _ = run_epoch(model, train_loader, opt, device, True)
        seen += len(train_loader.dataset)
        va, vp, vy = run_epoch(model, val_loader, opt, device, False)
        pred = (vp >= 0.5).astype(np.int64)
        f1 = f1_score(vy, pred, average="macro")
        hist.append({"epoch": epoch, "train_loss": tr, "val_loss": va, "val_f1": float(f1)})
        log.info("model=%s epoch=%s train_loss=%.4f val_loss=%.4f val_f1=%.4f", name, epoch, tr, va, f1)
        if va < best - 1e-4:
            best, stall = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
            if stall >= patience:
                log.info("early_stop model=%s epoch=%s", name, epoch)
                break
    model.load_state_dict(best_state)
    seconds = time.time() - t0
    return model, hist, seconds, seen / max(1e-6, seconds)


def save_model_plots(name, y, p, pred, out: Path) -> None:
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["gold 0", "gold 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    ax.set_title(name)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out / f"{name}_confusion.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots()
    bins = np.linspace(0, 1, 11)
    centers, accs, counts = [], [], []
    for i in range(10):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < 9 else p <= bins[i + 1])
        if not np.any(m):
            continue
        centers.append((bins[i] + bins[i + 1]) / 2)
        accs.append(y[m].mean())
        counts.append(int(m.sum()))
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(centers, accs, marker="o")
    ax.set_xlabel("predicted P(positive)")
    ax.set_ylabel("fraction positive")
    ax.set_title(f"{name} reliability")
    fig.tight_layout()
    fig.savefig(out / f"{name}_reliability.png", dpi=120)
    plt.close(fig)


def error_dump(model, texts, x, y, lengths, device, itos, path, n=20):
    model.eval()
    rows = []
    with torch.no_grad():
        xb = torch.from_numpy(x.astype(np.int64)).to(device)
        lb = torch.from_numpy(lengths.astype(np.int64)).to(device)
        # chunk so the whole test set fits
        probs = []
        attn = []
        for s in range(0, len(y), 256):
            sl = slice(s, s + 256)
            out = model(xb[sl], lb[sl], return_attn=True) if isinstance(model, BiLSTMAttn) else (model(xb[sl], lb[sl]), None)
            logit, w = out
            probs.append(torch.sigmoid(logit).cpu().numpy())
            if w is not None:
                attn.append(w.cpu().numpy())
        p = np.concatenate(probs)
        pred = (p >= 0.5).astype(np.int64)
        att = np.concatenate(attn) if attn else None
        used: set[int] = set()

        def take(cands, bucket):
            for i in cands:
                i = int(i)
                if i in used:
                    continue
                used.add(i)
                item = {
                    "bucket": bucket,
                    "text": " ".join(texts[i])[:500],
                    "gold": int(y[i]),
                    "pred": int(pred[i]),
                    "p_pos": float(p[i]),
                }
                if att is not None:
                    order = np.argsort(-att[i, : lengths[i]])[:8]
                    item["top_attn"] = [itos[int(x[i, j])] for j in order]
                rows.append(item)
                if sum(1 for r in rows if r["bucket"] == bucket) == 5:
                    return

        fp = np.flatnonzero((pred == 1) & (y == 0))
        take(fp[np.argsort(-p[fp])], "confident_fp")
        fn = np.flatnonzero((pred == 0) & (y == 1))
        take(fn[np.argsort(p[fn])], "confident_fn")
        wrong = np.flatnonzero(pred != y)
        take(wrong[np.argsort(np.abs(p[wrong] - 0.5))], "near_threshold")
        long_wrong = np.array([i for i in wrong if len(texts[int(i)]) > 200], dtype=np.int64)
        if len(long_wrong):
            take(long_wrong[np.argsort(-np.abs(p[long_wrong] - 0.5))], "slice_long")
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return p, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ayush/base.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg, cfg_hash = load_config(args.config)
    set_seeds(cfg.seed)
    device = resolve_device(cfg.device)
    log = open_run_log(cfg, "task2", "yelp", cfg_hash)
    if device.type != "cuda":
        raise SystemExit(f"refusing to train task 2 on {device}")

    repo = REPO
    t2 = cfg.task2
    member = repo / "task2_sentiment" / cfg.member
    split_dir = repo / cfg.paths.task2_splits
    proc = member / "data_processed"
    ckpt = member / "checkpoints"
    out = member / "outputs"
    for d in (split_dir, proc, ckpt, out):
        d.mkdir(parents=True, exist_ok=True)

    n_train = 800 if args.smoke else int(t2.n_train)
    n_val = 200 if args.smoke else int(t2.n_val)
    epochs = 1 if args.smoke else int(t2.epochs)
    log.info("loading yelp n_train=%s n_val=%s smoke=%s", n_train, n_val, args.smoke)
    (train_text, train_y), (test_text, test_y) = load_yelp()
    assert set(np.unique(train_y)).issubset({0, 1})

    train_idx_path = split_dir / "train_idx.npy"
    val_idx_path = split_dir / "val_idx.npy"
    if args.smoke or not (train_idx_path.exists() and val_idx_path.exists()):
        tr_i, va_i = make_stratified_indices(train_y, n_train, n_val, int(t2.split_seed))
        if not args.smoke:
            np.save(train_idx_path, tr_i)
            np.save(val_idx_path, va_i)
            log.info("wrote splits %s %s", len(tr_i), len(va_i))
    else:
        tr_i, va_i = np.load(train_idx_path), np.load(val_idx_path)
    te_i = np.arange(len(test_y))
    if args.smoke:
        te_i = te_i[:200]

    lemma_fn = lemmatizer()
    # One-epoch TextCNN ablation: stopwords+lemma versus neither. Full training uses the better val loss.
    use_stop, use_lemma = True, True
    if not args.smoke:
        scores = {}
        for flag in (False, True):
            rows, keep, n_empty, n_dup = tokenize_rows(train_text, tr_i, flag, flag, lemma_fn)
            vrows, vkeep, _, _ = tokenize_rows(train_text, va_i, flag, flag, lemma_fn)
            stoi = build_vocab(rows, int(t2.max_vocab))
            x, ln = encode_rows(rows, stoi, int(t2.max_len))
            vx, vln = encode_rows(vrows, stoi, int(t2.max_len))
            y = train_y[keep]
            vy = train_y[vkeep]
            loader = make_loader(x, y, ln, int(t2.batch_size), True)
            vloader = make_loader(vx, vy, vln, int(t2.batch_size), False)
            model = TextCNN(len(stoi)).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=float(t2.lr))
            tr_loss, _, _ = run_epoch(model, loader, opt, device, True)
            va_loss, vp, vyg = run_epoch(model, vloader, opt, device, False)
            scores[flag] = va_loss
            log.info("ablation stop_lemma=%s empty=%s dup=%s train_loss=%.4f val_loss=%.4f", flag, n_empty, n_dup, tr_loss, va_loss)
        use_stop = use_lemma = scores[True] <= scores[False]
        log.info("preprocess_choice stop_and_lemma=%s scores=%s", use_stop, scores)

    def bundle(texts, labels, indices, drop_dups):
        rows, keep, n_empty, n_dup = tokenize_rows(texts, indices, use_stop, use_lemma, lemma_fn, drop_dups=drop_dups)
        return rows, labels[keep], keep, n_empty, n_dup

    tr_rows, tr_y, tr_keep, e1, d1 = bundle(train_text, train_y, tr_i, True)
    va_rows, va_y, va_keep, e2, d2 = bundle(train_text, train_y, va_i, True)
    te_rows, te_y, te_keep, e3, d3 = bundle(test_text, test_y, te_i, False)
    log.info("dropped empty=%s dup=%s test_rows=%s", e1 + e2 + e3, d1 + d2 + d3, len(te_y))
    if not args.smoke and len(te_y) != 38000:
        raise SystemExit(f"test set is {len(te_y)}, expected 38000")
    if not args.smoke:
        np.save(split_dir / "test_idx.npy", te_keep)
    stoi = build_vocab(tr_rows, int(t2.max_vocab))
    (proc / "vocab.json").write_text(json.dumps(stoi))
    itos = {i: s for s, i in stoi.items()}
    log.info("vocab=%s", len(stoi))

    def enc(rows):
        return encode_rows(rows, stoi, int(t2.max_len))

    tr_x, tr_l = enc(tr_rows)
    va_x, va_l = enc(va_rows)
    te_x, te_l = enc(te_rows)
    flags = slice_flags(te_rows)

    fig, ax = plt.subplots()
    ax.hist(tr_l, bins=40)
    ax.set_title("train token lengths")
    fig.tight_layout()
    fig.savefig(out / "length_hist.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.bar(["neg", "pos"], [int((train_y == 0).sum()), int((train_y == 1).sum())])
    ax.set_title("official train class counts")
    fig.tight_layout()
    fig.savefig(out / "class_official.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.bar(["train neg", "train pos", "val neg", "val pos"], [int((tr_y == 0).sum()), int((tr_y == 1).sum()), int((va_y == 0).sum()), int((va_y == 1).sum())])
    ax.set_title("split class counts")
    fig.tight_layout()
    fig.savefig(out / "class_split.png", dpi=120)
    plt.close(fig)

    train_loader = make_loader(tr_x, tr_y, tr_l, int(t2.batch_size), True)
    val_loader = make_loader(va_x, va_y, va_l, int(t2.batch_size), False)
    rows = []
    preds = {}
    probs = {}
    t_all = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for name in list(t2.models):
        model, hist, seconds, eps = fit(
            name, len(stoi), train_loader, val_loader, device, epochs, float(t2.lr), int(t2.early_stop_patience), log
        )
        torch.save({"model": model.state_dict(), "name": name, "vocab": len(stoi)}, ckpt / f"{name}.pt")
        (out / f"{name}_history.json").write_text(json.dumps(hist))
        p, pred = error_dump(model, te_rows, te_x, te_y, te_l, device, itos, out / f"{name}_errors.jsonl")
        preds[name] = pred
        probs[name] = p
        save_model_plots(name, te_y, p, pred, out)
        n_params = sum(p_.numel() for p_ in model.parameters())
        rows.append(metric_row(name, te_y, p, flags, n_params, seconds, eps, peak_mem_gb()))
        log.info("tested model=%s acc=%.4f", name, rows[-1]["accuracy"])
        del model
        torch.cuda.empty_cache()

    pairs = {}
    names = list(preds)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairs[f"{a}_vs_{b}"] = mcnemar(te_y, preds[a], preds[b])
    (out / "mcnemar.json").write_text(json.dumps(pairs, indent=2))
    path = member / "metrics_report.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    write_manifest(cfg, 2, cfg_hash, checkpoint=str((ckpt / "textcnn.pt").relative_to(repo)), duration_sec=time.time() - t_all, metric_rows={"metrics_report.csv": len(rows)}, extra={"preprocess_stop_lemma": use_stop})
    log.info("done rows=%s", len(rows))


if __name__ == "__main__":
    main()
