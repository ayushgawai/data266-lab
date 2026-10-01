"""Shape checks for the three classifiers. No dataset."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from task2_sentiment.ayush.src.models import BiLSTMAttn, BiLSTMPool, TextCNN


def _batch():
    x = torch.randint(1, 50, (4, 16))
    x[:, 12:] = 0
    lengths = torch.tensor([16, 12, 8, 10])
    return x, lengths


def test_shapes():
    x, lengths = _batch()
    for m in (TextCNN(50), BiLSTMPool(50), BiLSTMAttn(50)):
        y = m(x, lengths)
        assert y.shape == (4,), y.shape
        y.sum().backward()


def test_pack_ignores_pad():
    x, lengths = _batch()
    m = BiLSTMPool(50)
    m.eval()
    a = m(x, lengths)
    x2 = x.clone()
    x2[:, 12:] = 7
    b = m(x2, lengths)
    # row 1 length is 12, so positions 12: must not change its logit
    assert torch.allclose(a[1], b[1], atol=1e-5)


def test_attn_sums_to_one_on_valid():
    x, lengths = _batch()
    m = BiLSTMAttn(50)
    m.eval()
    _, w = m(x, lengths, return_attn=True)
    assert torch.allclose(w.sum(-1), torch.ones(4), atol=1e-5)
    assert torch.all(w[1, 12:] <= 1e-6)


if __name__ == "__main__":
    test_shapes()
    test_pack_ignores_pad()
    test_attn_sums_to_one_on_valid()
    print("task2 checks ok")
