"""Shape, causal mask, and schedule checks. CPU only, no data download."""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from task1_llm.ayush.src.model import CausalSelfAttention, CharGPT, count_parameters
from task1_llm.ayush.src.train import cosine_lr, ngram_stats


def test_mask_blocks_future():
    attn = CausalSelfAttention(32, 4, 0.0, 16)
    attn.eval()
    x = torch.randn(2, 16, 32)
    y1 = attn(x)
    x2 = x.clone()
    x2[:, 10:] = torch.randn(2, 6, 32)
    y2 = attn(x2)
    assert torch.allclose(y1[:, :10], y2[:, :10], atol=1e-5), "future tokens changed past outputs"


def test_shapes_and_tying():
    m = CharGPT(40, block_size=16, n_layer=2, n_head=4, d_model=32, d_ff=64, dropout=0.0)
    logits = m(torch.randint(0, 40, (3, 16)))
    assert logits.shape == (3, 16, 40)
    assert m.head.weight.data_ptr() == m.tok_emb.weight.data_ptr()
    assert count_parameters(m) > 0
    try:
        m(torch.randint(0, 40, (1, 17)))
        raise AssertionError("should reject T > block_size")
    except ValueError:
        pass


def test_fixed_window_is_stable():
    from task1_llm.ayush.src.data import FixedWindowDataset
    stream = list(range(1000))
    ds = FixedWindowDataset(stream, 8, 4)
    x0, y0 = ds[0]
    x0b, _ = ds[0]
    assert torch.equal(x0, x0b)
    assert x0[0].item() == 0 and y0[0].item() == 1
    assert ds[1][0][0].item() == 8


def test_schedule():
    assert abs(cosine_lr(0, 200, 1000, 3e-4) - 3e-4 / 200) < 1e-12
    assert abs(cosine_lr(199, 200, 1000, 3e-4) - 3e-4) < 1e-12
    assert abs(cosine_lr(1000, 200, 1000, 3e-4) - 3e-5) < 1e-12


def test_diversity():
    stats = ngram_stats(["abababab", "cccc"])
    assert stats["distinct_1"] < 1.0
    assert stats["rep_4gram"] > 0.5


def test_one_backward():
    m = CharGPT(20, block_size=8, n_layer=2, n_head=4, d_model=32, d_ff=64, dropout=0.0)
    x = torch.randint(0, 20, (4, 8))
    y = torch.randint(0, 20, (4, 8))
    loss = torch.nn.functional.cross_entropy(m(x).reshape(-1, 20), y.reshape(-1))
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    assert math.isfinite(float(loss)) and math.isfinite(float(gn))


if __name__ == "__main__":
    test_mask_blocks_future()
    test_shapes_and_tying()
    test_fixed_window_is_stable()
    test_schedule()
    test_diversity()
    test_one_backward()
    print("task1 checks ok")
