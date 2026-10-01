"""Tests for the shared Task 1 metrics and the runtime contract."""
from __future__ import annotations

import math

import pytest
import torch

from common import runtime as rt
from common.metrics_lm import (bits_per_char, distinct_n, evaluate_lm, grad_norm_stats,
                               perplexity, repeated_4gram_rate, spike_count)


def test_perplexity_and_bpc():
    assert perplexity(math.log(4)) == pytest.approx(4.0)
    assert bits_per_char(math.log(2)) == pytest.approx(1.0)


def test_distinct_n_pooled():
    assert distinct_n(["a b c", "a b d"], 1) == pytest.approx(4 / 6)
    assert distinct_n(["a b c", "a b d"], 2) == pytest.approx(3 / 4)
    assert distinct_n(["a"], 3) == 0.0


def test_repeated_4gram_rate():
    # 4-grams: abcd bcda cdab dabc abcd -> "abcd" occurs twice: 2 of 5 occurrences
    assert repeated_4gram_rate(["a b c d a b c d"]) == pytest.approx(2 / 5)
    assert repeated_4gram_rate(["a b c d e f g"]) == 0.0
    assert repeated_4gram_rate(["x x x x x x"]) == 1.0
    # averaged per sample, so one looping sample does not hide behind a long clean one
    assert repeated_4gram_rate(["x x x x x x", "a b c d e f g"]) == pytest.approx(0.5)


def test_spikes_and_grad_stats():
    vals = [1.0] * 20 + [6.0] + [1.0] * 5 + [float("nan")]
    assert spike_count(vals, factor=5.0) == 1
    s = grad_norm_stats(vals)
    assert s["max"] == 6.0 and s["median"] == 1.0 and s["spikes"] == 1 and s["nan_count"] == 1


def test_evaluate_lm_uniform_model():
    V = 7

    class Uniform(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(*x.shape, V)

    batches = [(torch.zeros(2, 5, dtype=torch.long), torch.zeros(2, 5, dtype=torch.long))]
    r = evaluate_lm(Uniform(), batches, torch.device("cpu"))
    assert r["ce"] == pytest.approx(math.log(V)) and r["n_tokens"] == 10
    assert r["acc"] == 1.0          # argmax of all-equal logits is index 0 == target 0


def test_config_inheritance_and_hash(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nnest: {x: 1, y: 2}\n")
    (tmp_path / "child.yaml").write_text("inherits: base.yaml\nnest: {y: 3}\n")
    cfg, h = rt.load_config(tmp_path / "child.yaml")
    assert cfg.a == 1 and cfg.nest.x == 1 and cfg.nest.y == 3
    assert rt.load_config(tmp_path / "child.yaml")[1] == h
    (tmp_path / "base.yaml").write_text("a: 2\nnest: {x: 1, y: 2}\n")
    assert rt.load_config(tmp_path / "child.yaml")[1] != h   # parent change -> new hash


def test_run_log_never_overwrites(tmp_path, monkeypatch):
    cfg = rt.DotDict({"paths": {"logs": str(tmp_path)}, "run_name": "t", "device": "cpu"})
    monkeypatch.setattr(rt.time, "strftime", lambda fmt, _=None: "20260101T000000Z")
    logger, path = rt.open_run_log(cfg, "hash", "task1", "unit")
    rt.close_run_log(logger)
    assert path.read_text().splitlines()[0].endswith("config_sha256 hash")
    with pytest.raises(FileExistsError):
        rt.open_run_log(cfg, "hash", "task1", "unit")


def test_rel_paths_are_repo_relative():
    root = rt.find_repo_root()
    assert rt.rel(root / "common" / "runtime.py") == "common/runtime.py"
    assert not rt.rel(root / "common").startswith("/")


# arch list of the kind a CUDA 12.x torch 2.4 build reports (no sm_89: the 4090 runs the sm_86 binaries)
ARCHS_241 = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]


@pytest.mark.parametrize("cap,archs,ok", [
    ((8, 9), ARCHS_241, True),                     # RTX 4090
    ((12, 0), ARCHS_241, False),                   # RTX 5090 on torch 2.4.1
    ((12, 0), ARCHS_241 + ["compute_90"], True),   # PTX fallback would JIT
    ((12, 0), ARCHS_241 + ["sm_120"], True),       # a build that supports Blackwell
])
def test_cuda_arch_check(monkeypatch, cap, archs, ok):
    monkeypatch.setattr(rt.torch.cuda, "get_device_capability", lambda d=None: cap)
    monkeypatch.setattr(rt.torch.cuda, "get_arch_list", lambda: archs)
    monkeypatch.setattr(rt.torch.cuda, "get_device_name", lambda d=None: "test GPU")
    if ok:
        rt.check_cuda_arch(rt.torch.device("cuda"))
    else:
        with pytest.raises(RuntimeError, match="not supported"):
            rt.check_cuda_arch(rt.torch.device("cuda"))


def test_cuda_request_never_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(rt.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA GPU"):
        rt.get_device("cuda")
    assert rt.get_device("auto").type == "cpu"
    assert rt.get_device("cpu").type == "cpu"
    with pytest.raises(ValueError):
        rt.get_device("gpu")
