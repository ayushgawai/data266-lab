"""Pack every result into one zip, to upload to Google Drive before leaving the lab machine.

    python pack_results.py                 # results, logs, manifests, checkpoints, saved notebooks
    python pack_results.py --with-resume   # also the resume checkpoints (*_last.pt, ~2 GB for Task 3),
                                           # needed only if a run is unfinished and continues elsewhere

The lab wipes sessions after use and allows no USB drives (HPC Lab Rules 6 and 7), so this
zip is the backup. It contains each task's member folder (src/ with the saved notebooks,
config.yaml, outputs/, checkpoints/, metrics, write-ups), the shared split files and image
lists, the human-audit folder, and reproducibility/ (raw logs and manifests). It never
contains raw datasets or smoke-test outputs.
"""
from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MEMBER = "sneha"
INCLUDE = [f"task1_llm/{MEMBER}", f"task2_sentiment/{MEMBER}", f"task3_gan/{MEMBER}",
           "task2_sentiment/data/splits", "task3_gan/audit", "reproducibility"]
INCLUDE_FILES = ["task3_gan/data/*.txt", "task3_gan/data/lists_info.json"]


def wanted(p: Path, with_resume: bool) -> bool:
    parts = p.relative_to(ROOT).parts
    if "_smoke" in parts or "__pycache__" in parts or ".ipynb_checkpoints" in parts:
        return False
    if p.name.endswith("_last.pt") and not with_resume:
        return False
    return p.is_file() and p.name != ".DS_Store"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--with-resume", action="store_true", help="include *_last.pt resume checkpoints")
    args = ap.parse_args()
    files = []
    for d in INCLUDE:
        if (ROOT / d).exists():
            files += [p for p in sorted((ROOT / d).rglob("*")) if wanted(p, args.with_resume)]
    for pattern in INCLUDE_FILES:
        files += [p for p in sorted(ROOT.glob(pattern)) if wanted(p, args.with_resume)]
    out = ROOT / f"{MEMBER}_results_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for p in files:
            # checkpoints are already compressed binary: store them, don't deflate (much faster)
            zf.write(p, p.relative_to(ROOT).as_posix(),
                     compress_type=zipfile.ZIP_STORED if p.suffix in (".pt", ".npy", ".jpg", ".png") else None)
    big = [p for p in files if p.stat().st_size > 100e6]
    print(f"wrote {out.name}: {len(files)} files, {out.stat().st_size / 1e6:,.0f} MB")
    if big:
        print("over GitHub's 100 MB limit (keep these on Drive, link them in the manifest / results.md):")
        for p in big:
            print(f"  {p.relative_to(ROOT).as_posix()}  {p.stat().st_size / 1e6:,.0f} MB")
    print("Now upload this zip to Google Drive, check it opens, and only then delete your Desktop folder.")


if __name__ == "__main__":
    main()
