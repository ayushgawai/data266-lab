"""Character-level TinyStories loading. 100K train / 10K val stories (HF rows)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_tinystories_stories(
    n_train: int,
    n_val: int,
    split_seed: int,
    cache_dir: Path,
    hf_name: str = "roneneldan/TinyStories",
) -> tuple[list[str], list[str]]:
    """Load story lists. Prefer HF datasets (row = story). Cache plain JSONL locally."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path = cache_dir / f"train_{n_train}_seed{split_seed}.jsonl"
    val_path = cache_dir / f"val_{n_val}_seed{split_seed}.jsonl"

    if train_path.exists() and val_path.exists():
        train = [json.loads(l)["text"] for l in train_path.read_text().splitlines() if l.strip()]
        val = [json.loads(l)["text"] for l in val_path.read_text().splitlines() if l.strip()]
        return train, val

    from datasets import load_dataset

    # Full HF train split, then sample our own 100K/10K with member split_seed.
    ds = load_dataset(hf_name, split="train")
    rng = np.random.default_rng(split_seed)
    n_need = n_train + n_val
    if len(ds) < n_need:
        raise ValueError(f"Dataset has {len(ds)} stories, need {n_need}")
    idx = rng.choice(len(ds), size=n_need, replace=False)
    picked = ds.select([int(i) for i in idx])
    texts = picked["text"]
    train = list(texts[:n_train])
    val = list(texts[n_train:])

    with train_path.open("w") as f:
        for t in train:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    with val_path.open("w") as f:
        for t in val:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    return train, val


def build_char_vocab(stories: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted({c for s in stories for c in s})
    # Keep a simple vocab from the training slice only (brief requirement).
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int], unk: int | None = None) -> list[int]:
    if unk is None:
        return [stoi[c] for c in text if c in stoi]
    return [stoi.get(c, unk) for c in text]


def stories_to_stream(stories: list[str], stoi: dict[str, int]) -> np.ndarray:
    """Concatenate stories with a newline separator into one int stream."""
    parts = []
    for s in stories:
        ids = encode(s, stoi)
        if ids:
            parts.append(ids)
            # separator between stories if newline is in vocab
            if "\n" in stoi:
                parts.append([stoi["\n"]])
    if not parts:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate([np.asarray(p, dtype=np.int64) for p in parts])


class FixedWindowDataset(Dataset):
    """Non-overlapping character windows. The same index always returns the same pair."""

    def __init__(self, stream: np.ndarray, block_size: int, n_sequences: int):
        self.stream = torch.from_numpy(np.asarray(stream, dtype=np.int64))
        self.block_size = int(block_size)
        self.n_sequences = int(n_sequences)
        need = self.n_sequences * self.block_size + 1
        if len(self.stream) < need:
            raise ValueError(f"stream length {len(self.stream)} < {need} needed for {self.n_sequences} sequences")

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, index: int):
        start = int(index) * self.block_size
        x = self.stream[start : start + self.block_size]
        y = self.stream[start + 1 : start + 1 + self.block_size]
        return x, y


class CharLMDataset(Dataset):
    """Random character windows. __len__ is samples-per-epoch, not every offset."""

    def __init__(self, stream: np.ndarray, block_size: int, n_samples: int):
        self.stream = torch.from_numpy(stream.astype(np.int64))
        self.block_size = block_size
        self.n_samples = int(n_samples)
        if len(self.stream) <= block_size + 1:
            raise ValueError("Stream shorter than block_size")
        self.max_start = len(self.stream) - self.block_size - 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, i: int):
        # Deterministic mix of index + random so shuffle still helps.
        start = int((i * 9973 + torch.randint(0, self.max_start, (1,)).item()) % self.max_start)
        x = self.stream[start : start + self.block_size]
        y = self.stream[start + 1 : start + 1 + self.block_size]
        return x, y
