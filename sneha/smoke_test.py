"""One-command smoke test (brief, section 5): run a task notebook end to end in SMOKE mode.

    python smoke_test.py              # Task 1 (default): tiny GPT on the 22 MB TinyStories validation file
    python smoke_test.py --task 2     # Task 2: a few thousand Yelp reviews, 1 epoch
    python smoke_test.py --task 3     # Task 3: 2 tiny CycleGAN epochs on 16 images
    python smoke_test.py --task all

It sets DATA266_SMOKE=1, which each notebook's first cell reads, then executes every code
cell in order with IPython: no Jupyter server is started. Smoke outputs go to
task*/sneha/_smoke/ (git-ignored) and never touch the real run. Exits non-zero if any cell
fails. Each task downloads the data it needs on first use (Task 3 needs
task3_gan/data/dataset.zip; see the README).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = {
    "1": ROOT / "task1_llm" / "sneha" / "src" / "task1_llm_sneha.ipynb",
    "2": ROOT / "task2_sentiment" / "sneha" / "src" / "task2_sentiment_sneha.ipynb",
    "3": ROOT / "task3_gan" / "sneha" / "src" / "task3_gan_sneha.ipynb",
}


def run_notebook(path: Path) -> bool:
    from IPython.core.interactiveshell import InteractiveShell

    cells = [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    shell = InteractiveShell.instance()
    t0 = time.perf_counter()
    for i, cell in enumerate(cells, 1):
        print(f"\n----- {path.name}: code cell {i}/{len(cells)}", flush=True)
        if not shell.run_cell("".join(cell["source"])).success:
            print(f"\nSMOKE TEST FAILED in {path.name}, code cell {i}", flush=True)
            return False
    print(f"\nSMOKE TEST PASSED: {path.name} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", choices=["1", "2", "3", "all"], default="1")
    args = ap.parse_args()
    os.environ["DATA266_SMOKE"] = "1"
    os.environ.setdefault("MPLBACKEND", "Agg")      # headless: figures are saved, not shown
    os.chdir(ROOT)
    tasks = ["1", "2", "3"] if args.task == "all" else [args.task]
    if len(tasks) > 1:                              # one interpreter per notebook: no shared state
        import subprocess
        return max(subprocess.run([sys.executable, __file__, "--task", t]).returncode for t in tasks)
    return 0 if run_notebook(NOTEBOOKS[tasks[0]]) else 1


if __name__ == "__main__":
    sys.exit(main())
