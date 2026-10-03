"""Write 300 paired translations from the last CycleGAN checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common.runtime import load_config, resolve_device
from task3_gan.ayush.src.datasets import ImageFolderList, load_paths
from task3_gan.ayush.src.networks import ResnetGenerator


def save_jpg(tensor, path: Path) -> None:
    x = tensor[0].detach().cpu().clamp(-1, 1)
    arr = ((x + 1) * 0.5).permute(1, 2, 0).numpy()
    Image.fromarray((arr * 255).round().astype("uint8")).save(path, quality=95)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    cfg, _ = load_config("configs/ayush/base.yaml")
    device = resolve_device(cfg.device)
    member = REPO / "task3_gan" / cfg.member
    if args.checkpoint:
        best = Path(args.checkpoint)
        if not best.is_absolute():
            best = (REPO / best).resolve()
    else:
        ckpts = list((member / "checkpoints").glob("cyclegan_epoch*.pt"))
        if not ckpts:
            raise SystemExit(f"no checkpoint under {member / 'checkpoints'}")
        best = max(ckpts, key=lambda p: int(p.stem.split("epoch")[-1]))
    print("checkpoint", best)
    ckpt = torch.load(best, map_location=device, weights_only=False)
    g_ab, g_ba = ResnetGenerator().to(device), ResnetGenerator().to(device)
    g_ab.load_state_dict(ckpt.get("ema_g_ab", ckpt["g_ab"]))
    g_ba.load_state_dict(ckpt.get("ema_g_ba", ckpt["g_ba"]))
    g_ab.eval()
    g_ba.eval()
    data = REPO / cfg.paths.task3_data
    # The eval script sorts each folder and keeps the first 300. B2A has to be those photo stems.
    monets, _subset = load_paths(data, data / "photo_subset_2000.txt")
    ref_list = data / "ref_photos_300.txt"
    if ref_list.exists():
        photos = [data / "photo_jpg" / f"{line.strip()}.jpg" for line in ref_list.read_text().splitlines() if line.strip()]
    else:
        photos = sorted(p for p in (data / "photo_jpg").iterdir() if p.suffix.lower() == ".jpg" and not p.name.startswith("._"))[:300]
    monets = monets[:300]
    a_dir = member / "outputs" / "pred_A2B"
    b_dir = member / "outputs" / "pred_B2A"
    a_dir.mkdir(parents=True, exist_ok=True)
    b_dir.mkdir(parents=True, exist_ok=True)
    for old in list(a_dir.glob("*.jpg")) + list(b_dir.glob("*.jpg")):
        old.unlink()
    bs = max(1, args.batch_size)
    with torch.no_grad():
        for batch, stems in DataLoader(ImageFolderList(monets, train=False), batch_size=bs):
            out = g_ab(batch.to(device))
            for i, stem in enumerate(stems):
                save_jpg(out[i : i + 1], a_dir / f"{stem}.jpg")
        for batch, stems in DataLoader(ImageFolderList(photos, train=False), batch_size=bs):
            out = g_ba(batch.to(device))
            for i, stem in enumerate(stems):
                save_jpg(out[i : i + 1], b_dir / f"{stem}.jpg")
    print("wrote", len(list(a_dir.glob("*.jpg"))), "photos and", len(list(b_dir.glob("*.jpg"))), "monets")


if __name__ == "__main__":
    main()
