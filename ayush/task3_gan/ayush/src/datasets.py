"""Unpaired Monet and photo loading. Train augments. Inference does not."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _train_tf():
    return transforms.Compose([
        transforms.Resize(286, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def _test_tf():
    return transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class ImageFolderList(Dataset):
    def __init__(self, paths: list[Path], train: bool):
        self.paths = paths
        self.tf = _train_tf() if train else _test_tf()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        img = Image.open(path).convert("RGB")
        return self.tf(img), path.stem


def _jpgs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() == ".jpg" and not p.name.startswith("._"))


def write_photo_lists(data_dir: Path, seed: int = 42) -> None:
    """First 300 sorted names are the official FID reference and stay out of training."""
    photos = _jpgs(data_dir / "photo_jpg")
    if len(photos) < 900:
        raise ValueError(f"need at least 900 photos, found {len(photos)}")
    ref = photos[:300]
    rest = photos[300:]
    rng = random.Random(seed)
    rng.shuffle(rest)
    val = rest[:300]
    train = rest[300:]
    (data_dir / "ref_photos_300.txt").write_text("\n".join(p.stem for p in ref) + "\n")
    (data_dir / "val_photos_300.txt").write_text("\n".join(p.stem for p in val) + "\n")
    (data_dir / "photo_train.txt").write_text("\n".join(p.stem for p in train) + "\n")


def load_paths(data_dir: Path, photo_subset: Path, limit: int | None = None, all_photos: bool = False) -> tuple[list[Path], list[Path]]:
    monets = _jpgs(data_dir / "monet_jpg")
    photo_dir = data_dir / "photo_jpg"
    train_list = data_dir / "photo_train.txt"
    if all_photos and train_list.exists():
        wanted = [line.strip() for line in train_list.read_text().splitlines() if line.strip()]
        photos = [photo_dir / f"{stem}.jpg" for stem in wanted]
    elif all_photos:
        held = {line.strip() for line in (data_dir / "ref_photos_300.txt").read_text().splitlines()} if (data_dir / "ref_photos_300.txt").exists() else set()
        photos = [p for p in _jpgs(photo_dir) if p.stem not in held]
    else:
        wanted = [line.strip() for line in photo_subset.read_text().splitlines() if line.strip()]
        photos = []
        for stem in wanted:
            path = photo_dir / f"{stem}.jpg"
            if not path.exists():
                raise FileNotFoundError(path)
            photos.append(path)
    if limit is not None:
        monets = monets[:limit]
        photos = photos[:limit]
    return monets, photos
