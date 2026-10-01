"""Shared runtime helpers. Paths are always relative to the repo root."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml


def find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / "requirements.txt").exists():
            return p
    raise FileNotFoundError("Could not find repo root (requirements.txt).")


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_config(path: str | Path) -> tuple[SimpleNamespace, str]:
    path = Path(path)
    if not path.is_absolute():
        path = find_repo_root() / path
    raw = path.read_text()
    cfg_hash = hashlib.sha256(raw.encode()).hexdigest()
    data = yaml.safe_load(raw)
    return _to_ns(data), cfg_hash


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def resolve_device(preferred: str = "cuda"):
    import torch

    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def peak_mem_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    return 0.0


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def open_run_log(cfg: SimpleNamespace, task: str, model: str, cfg_hash: str) -> logging.Logger:
    repo = find_repo_root()
    member = getattr(cfg, "member", "ayush")
    log_dir = repo / getattr(cfg.paths, "logs", f"reproducibility/raw_logs/{member}")
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = log_dir / f"{task}_{model}_{stamp}.log"
    if path.exists():
        raise FileExistsError(f"Log already exists: {path}")

    logger = logging.getLogger(f"lab1.{task}.{model}.{stamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fh = logging.FileHandler(path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)

    import torch

    device = resolve_device(getattr(cfg, "device", "cuda"))
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else device.type
    logger.info("config_sha256=%s", cfg_hash)
    logger.info("git_sha=%s", _git_sha(repo))
    logger.info("python=%s", sys.version.replace("\n", " "))
    logger.info("platform=%s", platform.platform())
    logger.info("device=%s gpu=%s", device, gpu_name)
    logger.info("cwd=%s repo=%s", Path.cwd(), repo)
    logger.info("config=%s", json.dumps(_ns_to_dict(cfg), indent=2, default=str))
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        logger.info("pip_freeze_begin\n%s\npip_freeze_end", freeze)
    except Exception as e:
        logger.info("pip_freeze_failed=%s", e)
    logger.info("log_path=%s", path)
    return logger


def write_manifest(
    cfg: SimpleNamespace,
    task_n: int,
    cfg_hash: str,
    *,
    checkpoint: str,
    duration_sec: float,
    metric_rows: dict,
    extra: dict | None = None,
) -> Path:
    repo = find_repo_root()
    member = getattr(cfg, "member", "ayush")
    out_dir = repo / getattr(cfg.paths, "manifests", "reproducibility/manifests")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{member}_task{task_n}.json"
    import torch

    device = resolve_device(getattr(cfg, "device", "cuda"))
    payload = {
        "member": member,
        "task": task_n,
        "config_sha256": cfg_hash,
        "git_sha": _git_sha(repo),
        "seed": cfg.seed,
        "python": sys.version,
        "torch": getattr(torch, "__version__", "unknown"),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "duration_sec": duration_sec,
        "checkpoint": checkpoint,
        "metric_rows": metric_rows,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2))
    return path


def _ns_to_dict(obj: Any) -> Any:
    if isinstance(obj, SimpleNamespace):
        return {k: _ns_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_ns_to_dict(v) for v in obj]
    return obj


def colab_or_local_setup(repo_hint: str = "ayush") -> Path:
    """Detect Colab vs local/lab and return repo root. No absolute personal paths."""
    try:
        import google.colab  # type: ignore  # noqa: F401

        in_colab = True
    except Exception:
        in_colab = False

    if in_colab:
        # Expect user to mount Drive or clone into /content
        candidates = [
            Path("/content") / repo_hint,
            Path("/content/repo"),
            Path.cwd(),
        ]
        for c in candidates:
            if (c / "requirements.txt").exists():
                os.chdir(c)
                return c.resolve()
        raise FileNotFoundError(
            "Colab: clone or upload the team repo so requirements.txt is under /content."
        )
    return find_repo_root()
