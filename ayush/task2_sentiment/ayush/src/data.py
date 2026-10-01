"""Yelp polarity splits and word-level preprocessing. No pretrained embeddings."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

NEGATION = {"not", "no", "never", "nor", "neither", "dont", "doesnt", "didnt", "cant", "wont"}
CONTRAST = {"but", "however", "although", "though", "yet", "whereas"}


def clean_tokens(text: str, stop: set[str] | None = None, lemmatize=None) -> list[str]:
    text = text.replace("\\n", " ").replace('\\"', '"').replace("''", "'")
    text = text.lower()
    # "n't" includes the n, so a raw replace turns "can't" into "ca not".
    text = text.replace("can't", "can not").replace("won't", "will not").replace("n't", " not ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = [t for t in text.split() if t]
    if stop:
        toks = [t for t in toks if t not in stop]
    if lemmatize is not None:
        toks = [lemmatize(t) for t in toks]
    return toks


def make_stratified_indices(labels: np.ndarray, n_train: int, n_val: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal counts per class. Drawn from the official train labels only."""
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    classes = np.unique(labels)
    n_tr = n_train // len(classes)
    n_va = n_val // len(classes)
    for c in classes:
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        if len(idx) < n_tr + n_va:
            raise ValueError(f"class {c} has {len(idx)} rows, need {n_tr + n_va}")
        train_parts.append(idx[:n_tr])
        val_parts.append(idx[n_tr : n_tr + n_va])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def build_vocab(token_rows: list[list[str]], max_vocab: int) -> dict[str, int]:
    from collections import Counter

    counts = Counter(t for row in token_rows for t in row)
    stoi = {"<pad>": 0, "<unk>": 1}
    for word, _ in counts.most_common(max_vocab - 2):
        stoi[word] = len(stoi)
    return stoi


def encode_rows(token_rows: list[list[str]], stoi: dict[str, int], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    unk = stoi["<unk>"]
    x = np.zeros((len(token_rows), max_len), dtype=np.int32)
    lengths = np.zeros(len(token_rows), dtype=np.int32)
    for i, toks in enumerate(token_rows):
        ids = [stoi.get(t, unk) for t in toks[:max_len]]
        if not ids:
            ids = [unk]
        lengths[i] = len(ids)
        x[i, : len(ids)] = ids
    return x, lengths


def slice_flags(token_rows: list[list[str]]) -> dict[str, np.ndarray]:
    short, long, neg, con = [], [], [], []
    for toks in token_rows:
        n = len(toks)
        short.append(n < 50)
        long.append(n > 200)
        s = set(toks)
        neg.append(bool(s & NEGATION))
        con.append(bool(s & CONTRAST))
    return {k: np.asarray(v) for k, v in {"short": short, "long": long, "negation": neg, "contrast": con}.items()}


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj))
