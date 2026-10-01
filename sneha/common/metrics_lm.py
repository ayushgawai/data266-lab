"""Task 1 language-model metrics, shared by both members so the numbers are comparable.

Pure functions: no paths, no global state. Definitions (quote these in the report):

* cross entropy   mean next-character CE in nats over every target position
* perplexity      exp(CE)
* bits per char   CE / ln 2
* top-1 accuracy  fraction of positions where argmax(logits) equals the target
* distinct-n      unique word n-grams / total word n-grams, pooled over all
                  generated samples (corpus-level diversity, Li et al. 2016)
* repeated 4-gram rate
                  per sample: the share of word 4-gram occurrences whose 4-gram
                  appears more than once in that same sample; then the mean over
                  samples (within-sample degeneration)
* spikes          a value above `factor` x the median of the preceding `window`
                  finite values (counted once at least `min_history` exist)

Word n-grams are whitespace tokens. For a character-level model, character
distinct-1 is bounded by vocab size / length and says little about diversity.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_lm(model: torch.nn.Module, loader: Iterable, device: torch.device) -> dict:
    """One pass over `loader` of (x, y) batches: mean CE (nats) and top-1 accuracy.

    Runs in eval mode (dropout off) and restores the previous mode afterwards.
    """
    was_training = model.training
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                      reduction="sum").item()
        correct += (logits.argmax(dim=-1) == y).sum().item()
        count += y.numel()
    model.train(was_training)
    if count == 0:
        raise ValueError("evaluate_lm got an empty loader")
    return {"ce": total_loss / count, "acc": correct / count, "n_tokens": count}


def cross_entropy_eval(model, loader, device) -> float:
    return evaluate_lm(model, loader, device)["ce"]


def top1_accuracy(model, loader, device) -> float:
    return evaluate_lm(model, loader, device)["acc"]


def perplexity(ce: float) -> float:
    return math.exp(ce)


def bits_per_char(ce: float) -> float:
    return ce / math.log(2)


def _ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: Sequence[str], n: int) -> float:
    """Unique word n-grams / total word n-grams, pooled over all texts."""
    grams = [g for t in texts for g in _ngrams(t.split(), n)]
    return len(set(grams)) / len(grams) if grams else 0.0


def repeated_4gram_rate(texts: Sequence[str]) -> float:
    """Mean over texts of: share of word 4-gram occurrences whose 4-gram occurs
    more than once within the same text. 0.0 = no repetition, 1.0 = pure loops."""
    rates = []
    for t in texts:
        grams = _ngrams(t.split(), 4)
        if not grams:
            continue
        counts = Counter(grams)
        rates.append(sum(c for c in counts.values() if c > 1) / len(grams))
    return float(np.mean(rates)) if rates else 0.0


def spike_count(values: Sequence[float], factor: float, window: int = 100,
                min_history: int = 10) -> int:
    """Count values above `factor` x the median of the preceding `window` finite values."""
    history: list[float] = []
    spikes = 0
    for v in values:
        if not math.isfinite(v):
            continue
        if len(history) >= min_history and v > factor * float(np.median(history[-window:])):
            spikes += 1
        history.append(v)
    return spikes


def grad_norm_stats(norms: Sequence[float], spike_factor: float = 5.0, window: int = 100) -> dict:
    """max / median over finite pre-clip norms, spike count, and non-finite count."""
    finite = [v for v in norms if math.isfinite(v)]
    return {
        "max": max(finite) if finite else float("nan"),
        "median": float(np.median(finite)) if finite else float("nan"),
        "spikes": spike_count(norms, spike_factor, window),
        "nan_count": len(norms) - len(finite),
    }
