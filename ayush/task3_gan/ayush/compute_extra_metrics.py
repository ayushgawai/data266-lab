"""KID / density-coverage / content cosine / LPIPS / cycle L1 for Ayush epoch-100.
Matches Sneha's notebook definitions (Inception features, polynomial KID, k=5 density/coverage).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image

# File lives at task3_gan/ayush/compute_extra_metrics.py → project root is parents[2].
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from task3_gan.ayush.src.networks import ResnetGenerator

MEMBER = REPO / "task3_gan" / "ayush"
DATA = REPO / "task3_gan" / "data"
PRED_A2B = MEMBER / "outputs" / "pred_A2B"
PRED_B2A = MEMBER / "outputs" / "pred_B2A"
CKPT = MEMBER / "checkpoints" / "cyclegan_epoch100.pt"
OUT_CSV = MEMBER / "full_metrics_report.csv"
OUT_JSON = MEMBER / "full_metrics_report.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INCEPTION_TF = T.Compose([
    T.Resize(299), T.CenterCrop(299), T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
MODEL_TF = T.Compose([T.Resize(256), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

_inception = None


def list_jpgs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() == ".jpg" and not p.name.startswith("._"))


def inception():
    global _inception
    if _inception is None:
        m = tv_models.inception_v3(weights=tv_models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        m.fc = nn.Identity()
        _inception = m.to(DEVICE).eval()
    return _inception


@torch.no_grad()
def features(paths: list[Path], bs: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(paths), bs):
        batch = torch.stack([INCEPTION_TF(Image.open(p).convert("RGB")) for p in paths[i : i + bs]]).to(DEVICE)
        out.append(inception()(batch).float().cpu().numpy())
    return np.concatenate(out)


def kid(a: np.ndarray, b: np.ndarray, n_subsets: int = 100, subset_size: int = 100, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a, b = a.astype(np.float64), b.astype(np.float64)
    m = min(subset_size, len(a), len(b))
    d = a.shape[1]
    vals = []
    for _ in range(n_subsets):
        x = a[rng.choice(len(a), m, replace=False)]
        y = b[rng.choice(len(b), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def density_coverage(real: np.ndarray, fake: np.ndarray, k: int = 5) -> tuple[float, float]:
    def dist(x, y):
        return np.sqrt(np.maximum((x ** 2).sum(1)[:, None] + (y ** 2).sum(1)[None, :] - 2 * x @ y.T, 0))

    rr = dist(real, real)
    radii = np.sort(rr, axis=1)[:, k]
    rf = dist(real, fake)
    density = float((rf < radii[:, None]).sum() / (k * fake.shape[0]))
    coverage = float((rf.min(axis=1) < radii).mean())
    return density, coverage


def cosine_pairs(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return float((a * b).sum(1).mean())


def load_tensor(path: Path) -> torch.Tensor:
    return MODEL_TF(Image.open(path).convert("RGB"))


def main() -> None:
    t0 = time.time()
    print("device", DEVICE)
    monet_dir = DATA / "monet_jpg"
    photo_dir = DATA / "photo_jpg"
    a2b_paths = list_jpgs(PRED_A2B)
    b2a_paths = list_jpgs(PRED_B2A)
    if len(a2b_paths) < 300 or len(b2a_paths) < 300:
        raise SystemExit(f"need 300 preds each, got A2B={len(a2b_paths)} B2A={len(b2a_paths)}")

    monet_paths = [monet_dir / p.name for p in a2b_paths]
    # Same photo set as evaluate_local: first 300 sorted photos for A2B real domain.
    all_photos = list_jpgs(photo_dir)
    real_photo_paths = all_photos[:300]
    # B2A inputs are the stems in pred_B2A (ref photos).
    b2a_inputs = [photo_dir / p.name for p in b2a_paths]
    for p in monet_paths + b2a_inputs:
        if not p.exists():
            raise SystemExit(f"missing input {p}")

    print("extracting inception features...")
    f_monet = features(monet_paths)
    f_a2b = features(a2b_paths)
    f_photo = features(real_photo_paths)
    f_b2a = features(b2a_paths)
    f_b2a_in = features(b2a_inputs)

    r: dict = {}
    # Existing local FID/MiFID from submission
    sub = (MEMBER / "outputs" / "submission.csv").read_text().strip().splitlines()
    # fid_directions
    fid_txt = (MEMBER / "outputs" / "fid_directions.txt").read_text()
    vals = dict(line.split("=", 1) for line in fid_txt.splitlines() if "=" in line)
    r["fid_a2b"] = float(vals["fid_a2b"])
    r["fid_b2a"] = float(vals["fid_b2a"])
    r["fid_mean"] = (r["fid_a2b"] + r["fid_b2a"]) / 2
    r["mifid_a2b"] = float(vals["mifid_a2b"])
    r["mifid_b2a"] = float(vals["mifid_b2a"])
    r["mifid_mean"] = (r["mifid_a2b"] + r["mifid_b2a"]) / 2

    print("KID / density / content cosine...")
    r["kid_a2b_mean"], r["kid_a2b_std"] = kid(f_photo, f_a2b)
    r["kid_b2a_mean"], r["kid_b2a_std"] = kid(f_monet, f_b2a)
    r["density_a2b"], r["coverage_a2b"] = density_coverage(f_photo, f_a2b)
    r["density_b2a"], r["coverage_b2a"] = density_coverage(f_monet, f_b2a)
    r["content_cos_a2b"] = cosine_pairs(f_monet, f_a2b)
    r["content_cos_b2a"] = cosine_pairs(f_b2a_in, f_b2a)

    print("loading generators for cycle L1 + LPIPS...")
    try:
        import lpips
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lpips"])
        import lpips

    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    g_ab = ResnetGenerator().to(DEVICE).eval()
    g_ba = ResnetGenerator().to(DEVICE).eval()
    g_ab.load_state_dict(ck.get("ema_g_ab", ck["g_ab"]))
    g_ba.load_state_dict(ck.get("ema_g_ba", ck["g_ba"]))
    lp = lpips.LPIPS(net="alex", verbose=False).to(DEVICE).eval()

    cyc_a, cyc_b, lps_a2b, lps_b2a = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(monet_paths), 8):
            x = torch.stack([load_tensor(p) for p in monet_paths[i : i + 8]]).to(DEVICE)
            y = g_ab(x)
            cyc_a.append((g_ba(y) - x).abs().mean(dim=(1, 2, 3)).cpu())
            lps_a2b.append(lp(x, y).flatten().cpu())
        for i in range(0, len(b2a_inputs), 8):
            x = torch.stack([load_tensor(p) for p in b2a_inputs[i : i + 8]]).to(DEVICE)
            y = g_ba(x)
            cyc_b.append((g_ab(y) - x).abs().mean(dim=(1, 2, 3)).cpu())
            lps_b2a.append(lp(x, y).flatten().cpu())

    r["cycle_l1_a"] = float(torch.cat(cyc_a).mean())
    r["cycle_l1_b"] = float(torch.cat(cyc_b).mean())
    r["lpips_a2b"] = float(torch.cat(lps_a2b).mean())
    r["lpips_b2a"] = float(torch.cat(lps_b2a).mean())
    r["n_a2b"] = len(a2b_paths)
    r["n_b2a"] = len(b2a_paths)
    r["checkpoint"] = "task3_gan/ayush/checkpoints/cyclegan_epoch100.pt"
    r["seconds"] = time.time() - t0
    r["device"] = str(DEVICE)

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r.keys()))
        w.writeheader()
        w.writerow(r)
    OUT_JSON.write_text(json.dumps(r, indent=2))
    print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}, indent=2))
    print("wrote", OUT_CSV)


if __name__ == "__main__":
    main()
