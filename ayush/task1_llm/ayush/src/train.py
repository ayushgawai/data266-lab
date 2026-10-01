"""Train CharGPT on TinyStories (character LM)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Allow `python task1_llm/ayush/src/train.py` from repo root
REPO_CANDIDATE = Path(__file__).resolve().parents[3]
if str(REPO_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(REPO_CANDIDATE))

from common.runtime import (  # noqa: E402
    find_repo_root,
    load_config,
    open_run_log,
    peak_mem_gb,
    resolve_device,
    set_seeds,
    write_manifest,
)
from task1_llm.ayush.src.data import (  # noqa: E402
    FixedWindowDataset,
    build_char_vocab,
    load_tinystories_stories,
    stories_to_stream,
)
from task1_llm.ayush.src.model import CharGPT, count_parameters  # noqa: E402


def cosine_lr(step: int, warmup: int, total: int, base_lr: float, min_ratio: float = 0.1) -> float:
    """Warmup then cosine. Floor is 0.1 * base (3e-4 -> 3e-5), not zero."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


@torch.no_grad()
def eval_loss(model, loader, device) -> tuple[float, float]:
    """Token-weighted cross-entropy and top-1 next-character accuracy."""
    model.eval()
    total = 0.0
    correct = 0
    ntok = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        flat = logits.reshape(-1, logits.size(-1))
        yf = y.reshape(-1)
        total += F.cross_entropy(flat, yf, reduction="sum").item()
        correct += (flat.argmax(-1) == yf).sum().item()
        ntok += yf.numel()
    model.train()
    return total / max(1, ntok), correct / max(1, ntok)


def ngram_stats(texts: list[str]) -> dict[str, float]:
    """Character n-gram diversity on the generated samples."""
    from collections import Counter

    out = {}
    for n in (1, 2, 3):
        grams = [t[i : i + n] for t in texts for i in range(max(0, len(t) - n + 1))]
        out[f"distinct_{n}"] = (len(set(grams)) / len(grams)) if grams else 0.0
    four = [t[i : i + 4] for t in texts for i in range(max(0, len(t) - 3))]
    if not four:
        out["rep_4gram"] = 0.0
    else:
        counts = Counter(four)
        out["rep_4gram"] = sum(v for v in counts.values() if v > 1) / len(four)
    return out


def grad_summary(norms: list[float]) -> tuple[float, float, int, int]:
    arr = [v for v in norms if v == v]
    nans = sum(1 for v in norms if v != v)
    if not arr:
        return 0.0, 0.0, 0, nans
    seen: list[float] = []
    spikes = 0
    for v in arr:
        seen.append(v)
        med = sorted(seen)[len(seen) // 2]
        if med > 0 and v > 5.0 * med:
            spikes += 1
    mx = max(arr)
    med_all = sorted(arr)[len(arr) // 2]
    return mx, med_all, spikes, nans


def decode(ids, itos: dict[int, str]) -> str:
    return "".join(itos[int(i)] for i in ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ayush/base.yaml")
    ap.add_argument("--smoke", action="store_true", help="Tiny subset for path checks")
    args = ap.parse_args()

    repo = find_repo_root()
    cfg, cfg_hash = load_config(args.config)
    set_seeds(cfg.seed)
    device = resolve_device(cfg.device)
    log = open_run_log(cfg, "task1", "chargpt", cfg_hash)

    t1 = cfg.task1
    n_train = 2000 if args.smoke else int(t1.train_stories)
    n_val = 200 if args.smoke else int(t1.val_stories)
    epochs = 2 if args.smoke else int(t1.epochs)
    batch_size = 16 if args.smoke else int(t1.batch_size)

    data_dir = repo / cfg.paths.task1_data
    member_dir = repo / "task1_llm" / cfg.member
    ckpt_dir = member_dir / "checkpoints"
    out_dir = member_dir / "outputs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log.info("loading TinyStories stories train=%s val=%s", n_train, n_val)
    train_stories, val_stories = load_tinystories_stories(
        n_train=n_train,
        n_val=n_val,
        split_seed=int(t1.split_seed),
        cache_dir=data_dir / "cache",
        hf_name=t1.hf_dataset,
    )
    stoi, itos = build_char_vocab(train_stories)
    vocab_path = data_dir / "cache" / f"vocab_seed{t1.split_seed}.json"
    vocab_path.write_text(json.dumps({"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}))
    log.info("vocab_size=%s", len(stoi))

    train_stream = stories_to_stream(train_stories, stoi)
    val_stream = stories_to_stream(val_stories, stoi)
    log.info("train_chars=%s val_chars=%s", len(train_stream), len(val_stream))

    n_seq = 200 if args.smoke else int(getattr(t1, "train_sequences", 100000))
    n_vseq = 20 if args.smoke else int(getattr(t1, "val_sequences", 10000))
    block = int(t1.block_size)
    train_ds = FixedWindowDataset(train_stream, block, n_seq)
    val_ds = FixedWindowDataset(val_stream, block, n_vseq)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    log.info("train_sequences=%s val_sequences=%s steps_per_epoch=%s", n_seq, n_vseq, len(train_loader))

    model = CharGPT(
        vocab_size=len(stoi),
        block_size=int(t1.block_size),
        n_layer=int(t1.n_layer),
        n_head=int(t1.n_head),
        d_model=int(t1.d_model),
        d_ff=int(t1.d_ff),
        dropout=float(t1.dropout),
        tie_weights=bool(t1.tie_weights),
    ).to(device)
    n_params = count_parameters(model)
    log.info("params=%s device=%s", n_params, device)

    decay, nodecay = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else nodecay).append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(t1.weight_decay)},
            {"params": nodecay, "weight_decay": 0.0},
        ],
        lr=float(t1.lr),
        betas=(0.9, 0.95),
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    step = 0
    train_losses = []
    val_losses = []
    grad_norms: list[float] = []
    train_tokens = 0
    best_val = float("inf")
    best_path = ckpt_dir / "chargpt_best.pt"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for x, y in train_loader:
            step += 1
            lr = cosine_lr(step, int(t1.warmup_steps), total_steps, float(t1.lr))
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(t1.grad_clip))
            opt.step()
            running += loss.item()
            train_tokens += int(y.numel())
            grad_norms.append(float(grad_norm))
            if step % 50 == 0 or step == 1:
                log.info(
                    "step=%s epoch=%s loss=%.4f lr=%.6f grad_norm=%.4f mem_gb=%.3f",
                    step,
                    epoch,
                    loss.item(),
                    lr,
                    float(grad_norm),
                    peak_mem_gb(),
                )
        tr = running / max(1, steps_per_epoch)
        va, _ = eval_loss(model, val_loader, device)
        train_losses.append(tr)
        val_losses.append(va)
        log.info("epoch=%s train_loss=%.4f val_loss=%.4f", epoch, tr, va)
        if va < best_val:
            best_val = va
            torch.save(
                {
                    "model": model.state_dict(),
                    "stoi": stoi,
                    "itos": itos,
                    "config": {
                        "block_size": int(t1.block_size),
                        "n_layer": int(t1.n_layer),
                        "n_head": int(t1.n_head),
                        "d_model": int(t1.d_model),
                        "d_ff": int(t1.d_ff),
                        "dropout": float(t1.dropout),
                        "tie_weights": bool(t1.tie_weights),
                    },
                    "epoch": epoch,
                    "val_loss": va,
                },
                best_path,
            )
            log.info("saved_best %s val=%.4f", best_path, va)

    # curves
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, epochs + 1), train_losses, label="train")
    ax.plot(range(1, epochs + 1), val_losses, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.legend()
    ax.set_title("Task1 CharGPT loss")
    curve_path = out_dir / "loss_curves.png"
    fig.tight_layout()
    fig.savefig(curve_path, dpi=120)
    plt.close(fig)

    # generate samples
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    samples = []
    prompts = ["Once upon a time", "One day", "There was a", "The little", "A girl named"]
    n_gen = int(t1.gen.n_samples)
    gen_t0 = time.perf_counter()
    gen_new = 0
    for i in range(n_gen):
        p = prompts[i % len(prompts)]
        ids = [stoi[c] for c in p if c in stoi]
        if not ids:
            ids = [0]
        idx = torch.tensor([ids], device=device, dtype=torch.long)
        n_new = int(t1.gen.max_new_tokens)
        out = model.generate(idx, max_new_tokens=n_new, temperature=float(t1.gen.temperature))
        gen_new += n_new
        text = decode(out[0].tolist(), itos)
        samples.append(text)
        log.info("SAMPLE %s\n%s\n", i, text[:400])
    gen_sec = max(1e-6, time.perf_counter() - gen_t0)
    (out_dir / "generations.txt").write_text("\n\n----\n\n".join(samples))

    best_i = min(range(len(val_losses)), key=lambda i: val_losses[i])
    final_train = train_losses[best_i]
    final_val = val_losses[best_i]
    _, top1 = eval_loss(model, val_loader, device)
    ppl = math.exp(min(20.0, final_val))
    bpc = final_val / math.log(2)
    gap = final_val - final_train
    train_sec = max(1e-6, time.time() - t0)
    gmax, gmed, spikes, nans = grad_summary(grad_norms)
    div = ngram_stats(samples)
    metrics_path = member_dir / "metrics_report.csv"
    row = {
        "train_ce": final_train,
        "val_ce": final_val,
        "perplexity": ppl,
        "bits_per_char": bpc,
        "generalization_gap": gap,
        "top1_acc": top1,
        "distinct_1": div["distinct_1"],
        "distinct_2": div["distinct_2"],
        "distinct_3": div["distinct_3"],
        "rep_4gram": div["rep_4gram"],
        "grad_norm_max": gmax,
        "grad_norm_median": gmed,
        "grad_spikes": spikes,
        "grad_nans": nans,
        "param_count": n_params,
        "tokens_per_sec": train_tokens / train_sec,
        "gen_tokens_per_sec": gen_new / gen_sec,
        "peak_mem_gb": peak_mem_gb(),
        "train_time_sec": train_sec,
        "epochs": epochs,
        "train_stories": n_train,
        "val_stories": n_val,
        "vocab_size": len(stoi),
        "smoke": int(args.smoke),
    }
    with metrics_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    write_manifest(
        cfg,
        1,
        cfg_hash,
        checkpoint=str(best_path.relative_to(repo)),
        duration_sec=time.time() - t0,
        metric_rows={"metrics_report.csv": 1},
        extra={"curve": str(curve_path.relative_to(repo))},
    )
    log.info("done metrics=%s", row)


if __name__ == "__main__":
    main()
