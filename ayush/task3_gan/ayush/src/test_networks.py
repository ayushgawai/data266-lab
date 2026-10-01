"""Shape, loss formula, and one backward step. CPU, no images."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from task3_gan.ayush.src.networks import ImagePool, PatchDiscriminator, ResnetGenerator, init_weights


def test_batch_fills_5090():
    from task3_gan.ayush.src.train import choose_batch
    assert choose_batch(32, 0) == 8
    assert choose_batch(32, 16) == 16
    assert choose_batch(8, 0) == 1


def test_lr_stays_positive():
    from task3_gan.ayush.src.train import lr_factor
    assert lr_factor(0, 20, 10) == 1.0
    assert lr_factor(19, 20, 10) == 1.0
    assert abs(lr_factor(20, 20, 10) - (1 - 1 / 11)) < 1e-9
    assert lr_factor(29, 20, 10) > 0


def test_shapes():
    g = ResnetGenerator()
    d = PatchDiscriminator()
    x = torch.randn(1, 3, 256, 256)
    y = g(x)
    s = d(y)
    assert y.shape == (1, 3, 256, 256), y.shape
    assert s.shape == (1, 1, 30, 30), s.shape
    assert y.abs().max() <= 1 + 1e-4


def test_one_step_finite():
    g_ab, g_ba = ResnetGenerator(), ResnetGenerator()
    d_a, d_b = PatchDiscriminator(), PatchDiscriminator()
    for m in (g_ab, g_ba, d_a, d_b):
        init_weights(m)
    a = torch.randn(1, 3, 256, 256)
    b = torch.randn(1, 3, 256, 256)
    fake_b = g_ab(a)
    fake_a = g_ba(b)
    ones_b = torch.ones_like(d_b(fake_b))
    loss_g = F.mse_loss(d_b(fake_b), ones_b) + F.mse_loss(d_a(fake_a), torch.ones_like(d_a(fake_a)))
    loss_g = loss_g + 10 * (F.l1_loss(g_ba(fake_b), a) + F.l1_loss(g_ab(fake_a), b))
    loss_g = loss_g + 5 * (F.l1_loss(g_ab(b), b) + F.l1_loss(g_ba(a), a))
    loss_g.backward()
    assert torch.isfinite(loss_g).item()


def test_pool_returns_history():
    pool = ImagePool(2)
    a = torch.zeros(1, 1)
    b = torch.ones(1, 1)
    pool.query(a)
    pool.query(b)
    torch.manual_seed(0)
    got = pool.query(torch.full((1, 1), 2))
    assert got.shape == (1, 1)


if __name__ == "__main__":
    test_batch_fills_5090()
    test_lr_stays_positive()
    test_shapes()
    test_one_step_finite()
    test_pool_returns_history()
    print("task3 checks ok")
