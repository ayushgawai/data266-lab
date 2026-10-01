"""Shared runtime contract for DATA266 Lab 1, imported by every member's entry points.

Repo-root discovery, config loading and hashing, seeding, device selection, the
append-only run log, and the per-task manifest. Nothing here is member specific,
and every path written to disk is relative to the repo root, so no personal
absolute path ends up in a committed file.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
import yaml


class DotDict(dict):
    """A dict with recursive read-only attribute access: cfg.task1.block_size."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return DotDict(value) if isinstance(value, dict) else value


def find_repo_root(start: str | Path | None = None) -> Path:
    """Walk up from `start` (default: this file) to the folder holding requirements.txt."""
    p = Path(start or __file__).resolve()
    for cand in [p, *p.parents]:
        if (cand / "requirements.txt").is_file():
            return cand
    raise RuntimeError("repo root not found: no requirements.txt in any parent folder")


def resolve_path(p: str | Path) -> Path:
    """Config paths are relative to the repo root; absolute paths pass through unchanged."""
    p = Path(p)
    return p if p.is_absolute() else find_repo_root() / p


def rel(p: str | Path) -> str:
    """Repo-relative POSIX string for logs and manifests. Falls back to the plain path."""
    p = Path(p).resolve()
    try:
        return p.relative_to(find_repo_root()).as_posix()
    except ValueError:
        return p.as_posix()


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_yaml(path: Path, chain: tuple[Path, ...] = ()) -> dict:
    """Load YAML, following an optional `inherits: <file>` key relative to this file."""
    if path in chain:
        raise ValueError(f"config inheritance cycle at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("inherits", None)
    if parent is None:
        return raw
    return _deep_merge(_load_yaml((path.parent / parent).resolve(), chain + (path,)), raw)


def load_config(path: str | Path) -> tuple[DotDict, str]:
    """Return (config, sha256). The hash is over the fully resolved config, so an
    inherited value that changes also changes the hash of every child config."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = find_repo_root() / path
    cfg = _load_yaml(path.resolve())
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return DotDict(cfg), hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def to_plain(cfg: dict) -> dict:
    """DotDict -> plain nested dict, for YAML/JSON dumping."""
    return json.loads(json.dumps(cfg))


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested: str = "cuda") -> torch.device:
    """`cuda`: a supported GPU or an error, never a silent CPU fallback (a CPU-only
    torch wheel would otherwise turn a minutes-long GPU run into hours on the CPU).
    `auto`: the GPU if there is one, else the CPU. `cpu`: always the CPU."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested not in ("cuda", "auto"):
        raise ValueError(f"device must be cuda, auto or cpu, not {requested!r}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        check_cuda_arch(device)
        return device
    if requested == "auto":
        return torch.device("cpu")
    raise RuntimeError(
        f"device: cuda, but torch {torch.__version__} sees no CUDA GPU "
        f"(torch.version.cuda = {torch.version.cuda}). A CPU-only torch wheel reports "
        f"None there: reinstall with the README's CUDA command. To run on the CPU on "
        f"purpose, use a config with device: auto or cpu.")


def check_cuda_arch(device: torch.device) -> None:
    """Fail fast if this torch build has no kernels for the GPU, instead of failing at
    the first CUDA op mid-run. The team pins torch 2.4.1 and trains on RTX 4090s
    (sm_89, served by the sm_86 binaries); RTX 50-series cards (sm_120) need torch >= 2.7.

    CUDA compatibility rules: an sm_XY binary runs on any sm_XZ with Z >= Y (same major
    version), and compute_XY PTX can be JIT-compiled for any newer card.
    """
    major, minor = torch.cuda.get_device_capability(device)
    archs = torch.cuda.get_arch_list()
    if not archs:                      # build reports nothing; let torch decide
        return
    cap = 10 * major + minor
    ok = any(
        (a.startswith("sm_") and int(a[3:]) // 10 == major and int(a[3:]) <= cap)
        or (a.startswith("compute_") and int(a[8:]) <= cap)
        for a in archs)
    if not ok:
        raise RuntimeError(
            f"{torch.cuda.get_device_name(device)} (sm_{major}{minor}) is not supported by "
            f"torch {torch.__version__} (built for {', '.join(archs)}). Use an RTX 4090, or "
            f"a torch build that supports this card.")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return f"cpu ({platform.processor() or platform.machine()})"


def reset_peak_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_mem_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def sync(device: torch.device) -> None:
    """Wait for queued GPU work, so wall-clock timings are real."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def git_sha() -> str:
    """The checked-out commit. Reads .git directly when the git program is missing."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=find_repo_root(), text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        pass
    try:
        g = find_repo_root() / ".git"
        head = (g / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        if (g / ref).exists():
            return (g / ref).read_text().strip()
        packed = g / "packed-refs"
        for line in (packed.read_text().splitlines() if packed.exists() else []):
            if line.endswith(" " + ref):
                return line.split()[0]
    except Exception:
        pass
    return "unknown"


def installed_packages() -> list[str]:
    """`pip freeze` equivalent that also works in venvs created without pip."""
    return sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def open_run_log(cfg: DotDict, cfg_hash: str, task: str, model: str) -> tuple[logging.Logger, Path]:
    """Create a new timestamped log file and write the header block.

    The file is opened with mode "x", so an existing log can never be overwritten:
    raw logs are the evidence trail. Per-step lines go to the file at DEBUG level;
    the console only gets INFO lines.
    """
    log_dir = resolve_path(cfg.paths.logs)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = log_dir / f"{task}_{model}_{cfg.run_name}_{stamp}.log"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite an existing log: {rel(path)}")

    logger = logging.getLogger(f"{task}.{model}.{cfg.run_name}.{stamp}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, mode="x", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    for h in (file_handler, console):
        h.setFormatter(fmt)
        logger.addHandler(h)

    dev = get_device(cfg.device)
    logger.info(f"config_sha256 {cfg_hash}")  # first line, per the logging contract
    logger.info(f"git_sha {git_sha()}")
    logger.info(f"device {dev.type} | {device_name(dev)}")
    logger.info(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
                f"python {platform.python_version()} | platform {platform.platform()}")
    # full config and package list: file only (DEBUG), they would flood the console
    logger.debug("config " + json.dumps(to_plain(cfg), sort_keys=True, ensure_ascii=False))
    logger.debug("packages " + json.dumps(installed_packages()))
    logger.info(f"log_path {rel(path)}")
    return logger, path


def close_run_log(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)


def snapshot_config(cfg: DotDict, cfg_hash: str, task_dir: Path, log_path: Path) -> None:
    """Write task_dir/config_used.yaml and append the log path to task_dir/logs.md,
    so each member task folder holds its config and log pointers (brief checklist)."""
    task_dir.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(to_plain(cfg), sort_keys=False, allow_unicode=True)
    (task_dir / "config_used.yaml").write_text(
        f"# resolved config of the most recent run; sha256 {cfg_hash}\n{body}", encoding="utf-8")
    logs_md = task_dir / "logs.md"
    if not logs_md.exists():
        logs_md.write_text("# Raw logs for this task\n\nAppended automatically at the start "
                           "of every run. The logs themselves are never edited.\n\n",
                           encoding="utf-8")
    with open(logs_md, "a", encoding="utf-8") as f:
        f.write(f"- `{rel(log_path)}` (run `{cfg.run_name}`, config sha256 `{cfg_hash[:12]}`)\n")


def write_manifest(cfg: DotDict, cfg_hash: str, task: str, *, checkpoint: Path,
                   metric_rows: dict, duration_s: float, log_paths: list[Path],
                   extra: dict | None = None) -> Path:
    """Per-task manifest. `metric_rows` maps the checkpoint to the CSV rows it
    produced; that mapping is what makes each reported number traceable."""
    out_dir = resolve_path(cfg.paths.manifests)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = get_device(cfg.device)
    manifest = {
        "member": cfg.member,
        "task": task,
        "run_name": cfg.run_name,
        "config_sha256": cfg_hash,
        "git_sha": git_sha(),
        "seed": cfg.seed,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device_name(dev),
        "packages": installed_packages(),
        "duration_seconds": round(duration_s, 3),
        "checkpoint": {"path": rel(checkpoint), "sha256": sha256_file(checkpoint),
                       "bytes": Path(checkpoint).stat().st_size},
        "metric_rows": metric_rows,
        "raw_logs": [rel(p) for p in log_paths],
        "config": to_plain(cfg),
    }
    if extra:
        manifest.update(extra)
    name = f"{cfg.member}_{task}.json" if cfg.run_name == "main" else f"{cfg.member}_{task}_{cfg.run_name}.json"
    path = out_dir / name
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
