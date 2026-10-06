"""Score Sneha's Task 3 outputs with the PROVIDED evaluation script (brief 3.2.5), outside the notebook.

    cd sneha && python task3_gan/sneha/evaluate_local.py              # writes outputs/evaluate_local/submission.csv
    cd sneha && python task3_gan/sneha/evaluate_local.py --dry-run    # only shows the three replaced path lines

This is the notebook's run_provided_script (section 11 of src/task3_gan_sneha.ipynb) as a standalone file: it runs every code
cell of task3_gan/eval/Part3_Evaluation_Script.ipynb unchanged, except its BASE, GEN_A2B and GEN_B2A path lines, which are
pointed at task3_gan/data and this member's outputs/pred_A2B, outputs/pred_B2A. The script itself writes submission.csv.
By default that goes to outputs/evaluate_local/, so the submitted task3_gan/sneha/submission.csv is never overwritten.
Needs the course's images in task3_gan/data/monet_jpg and photo_jpg (the notebook extracts dataset.zip there).
"""
import argparse
import contextlib
import io
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent                       # task3_gan/sneha
ROOT = HERE.parents[1]                                       # sneha/
SCRIPT = ROOT / "task3_gan" / "eval" / "Part3_Evaluation_Script.ipynb"
DATA = ROOT / "task3_gan" / "data"
PRED_A2B, PRED_B2A = HERE / "outputs" / "pred_A2B", HERE / "outputs" / "pred_B2A"

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=HERE / "outputs" / "evaluate_local", help="folder for the script's submission.csv")
ap.add_argument("--dry-run", action="store_true", help="show the replaced path lines without running the script")
args = ap.parse_args()

subs = {r"^BASE\s*=.*$": f"BASE = {str(DATA)!r}",
        r"^GEN_A2B\s*=.*$": f"GEN_A2B    = {str(PRED_A2B)!r}",
        r"^GEN_B2A\s*=.*$": f"GEN_B2A    = {str(PRED_B2A)!r}"}
cells, done = [], set()
for cell in json.loads(SCRIPT.read_text(encoding="utf-8"))["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    for pat, rep in subs.items():
        new = re.sub(pat, rep, src, flags=re.M)
        if new != src:
            done.add(pat)
            print(f"provided script: replaced path line -> {rep}")
            src = new
    cells.append(src)
assert done == set(subs), f"could not find the path lines {set(subs) - done} in the provided script"
if args.dry_run:
    raise SystemExit(f"dry run: {len(cells)} code cells found, nothing executed")

for d in (DATA / "monet_jpg", DATA / "photo_jpg", PRED_A2B, PRED_B2A):
    assert d.is_dir(), f"missing folder {d}"
args.out.mkdir(parents=True, exist_ok=True)
ns, cwd = {}, os.getcwd()
os.chdir(args.out)                                           # the script writes submission.csv to the working directory
try:
    for src in cells:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(src, "Part3_Evaluation_Script", "exec"), ns)
        if buf.getvalue().strip():
            print(buf.getvalue().strip())
finally:
    os.chdir(cwd)
print({k: float(ns[k]) for k in ("fid_A2B", "mifid_A2B", "fid_B2A", "mifid_B2A", "sub_fid", "sub_mifid")})
print(f"wrote {args.out / 'submission.csv'} (compare with task3_gan/sneha/submission.csv)")
