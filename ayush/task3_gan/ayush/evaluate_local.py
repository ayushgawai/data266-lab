"""Official-style FID / MiFID from Part3_Evaluation_Script.ipynb."""

from __future__ import annotations

import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
MEMBER = Path(__file__).resolve().parent
N_EVAL = 300
BATCH_SIZE = 32


def list_images(folder: Path) -> list[str]:
    paths = []
    for ext in (".jpg", ".jpeg", ".png"):
        paths.extend(glob.glob(str(folder / f"*{ext}")))
        paths.extend(glob.glob(str(folder / f"*{ext.upper()}")))
    # AppleDouble sidecars sort before real names and would poison the first 300.
    paths = sorted({p for p in paths if not Path(p).name.startswith("._")})
    return paths


def take_n(paths: list[str], n: int | None) -> list[str]:
    return paths if n is None else paths[: min(n, len(paths))]


def get_inception_model(device: torch.device):
    inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    inception.fc = nn.Identity()
    inception.to(device)
    inception.eval()
    return inception


INCEPTION_TF = T.Compose([
    T.Resize(299),
    T.CenterCrop(299),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def load_batch(paths: list[str]) -> torch.Tensor:
    return torch.stack([INCEPTION_TF(Image.open(p).convert("RGB")) for p in paths], dim=0)


@torch.no_grad()
def get_activations(model, image_paths: list[str], device: torch.device, batch_size: int = 32) -> np.ndarray:
    feats = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Inception activations"):
        x = load_batch(image_paths[i : i + batch_size]).to(device)
        feats.append(model(x).detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def _sym_matrix_sqrt(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mat = 0.5 * (mat + mat.T)
    evals, evecs = torch.linalg.eigh(mat)
    evals = torch.clamp(evals, min=0.0)
    if not torch.isfinite(evals).all():
        mat = mat + torch.eye(mat.shape[0], dtype=mat.dtype) * eps
        evals, evecs = torch.linalg.eigh(mat)
        evals = torch.clamp(evals, min=0.0)
    return (evecs * torch.sqrt(evals)) @ evecs.T


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    # Same Frechet formula as the staff notebook. Symmetrized eigh sqrt replaces
    # scipy.linalg.sqrtm (Windows Application Control blocks that scipy DLL).
    mu1 = torch.as_tensor(mu1, dtype=torch.float64)
    mu2 = torch.as_tensor(mu2, dtype=torch.float64)
    sigma1 = torch.as_tensor(sigma1, dtype=torch.float64)
    sigma2 = torch.as_tensor(sigma2, dtype=torch.float64)
    diff = mu1 - mu2
    covmean = _sym_matrix_sqrt(sigma1 @ sigma2, eps=eps)
    return float(diff.dot(diff) + torch.trace(sigma1 + sigma2 - 2 * covmean))


def calculate_fid_mifid(real_paths, gen_paths, device, batch_size=32):
    real_paths = sorted(real_paths)
    gen_paths = sorted(gen_paths)
    n = min(len(real_paths), len(gen_paths))
    real_paths, gen_paths = real_paths[:n], gen_paths[:n]
    model = get_inception_model(device)
    real_act = get_activations(model, real_paths, device, batch_size=batch_size)
    gen_act = get_activations(model, gen_paths, device, batch_size=batch_size)
    mu_r, sig_r = real_act.mean(axis=0), np.cov(real_act, rowvar=False)
    mu_g, sig_g = gen_act.mean(axis=0), np.cov(gen_act, rowvar=False)
    fid = frechet_distance(mu_r, sig_r, mu_g, sig_g)
    # Mean cosine distance, same pairing rule as the staff notebook.
    ra = real_act / (np.linalg.norm(real_act, axis=1, keepdims=True) + 1e-12)
    ga = gen_act / (np.linalg.norm(gen_act, axis=1, keepdims=True) + 1e-12)
    mifid = float(np.mean(1.0 - np.sum(ra * ga, axis=1)))
    return fid, mifid


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = REPO / "task3_gan" / "data"
    real_monet = take_n(list_images(data / "monet_jpg"), N_EVAL)
    real_photo = take_n(list_images(data / "photo_jpg"), N_EVAL)
    gen_a2b = take_n(list_images(MEMBER / "outputs" / "pred_A2B"), N_EVAL)
    gen_b2a = take_n(list_images(MEMBER / "outputs" / "pred_B2A"), N_EVAL)
    for name, paths in (("real_monet", real_monet), ("real_photo", real_photo), ("pred_A2B", gen_a2b), ("pred_B2A", gen_b2a)):
        if len(paths) < N_EVAL:
            raise SystemExit(f"{name} has {len(paths)} images, need {N_EVAL}")
    print(f"device={device}")
    print(f"counts monet={len(real_monet)} photo={len(real_photo)} a2b={len(gen_a2b)} b2a={len(gen_b2a)}")
    fid_b2a, mifid_b2a = calculate_fid_mifid(real_monet, gen_b2a, device, BATCH_SIZE)
    print(f"[Photo->Monet] FID={fid_b2a:.6f} MiFID={mifid_b2a:.6f}")
    fid_a2b, mifid_a2b = calculate_fid_mifid(real_photo, gen_a2b, device, BATCH_SIZE)
    print(f"[Monet->Photo] FID={fid_a2b:.6f} MiFID={mifid_a2b:.6f}")
    avg_fid = (fid_a2b + fid_b2a) / 2
    avg_mifid = (mifid_a2b + mifid_b2a) / 2
    out = MEMBER / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fid_directions.txt").write_text(
        f"fid_a2b={fid_a2b}\nmifid_a2b={mifid_a2b}\nfid_b2a={fid_b2a}\nmifid_b2a={mifid_b2a}\n"
    )
    with (out / "submission.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ID", "FID", "MiFID"])
        w.writeheader()
        w.writerow({"ID": 1, "FID": float(avg_fid), "MiFID": float(avg_mifid)})
    print(f"average FID={avg_fid:.6f} MiFID={avg_mifid:.6f}")
    print(f"wrote {out / 'submission.csv'}")


if __name__ == "__main__":
    main()
