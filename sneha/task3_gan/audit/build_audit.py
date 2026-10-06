"""Build the blinded human-audit set for Task 3 (brief 3.2.6): 30 fixed samples, both members, both directions.

    python sneha/task3_gan/audit/build_audit.py --data <folder with monet_jpg/ and photo_jpg/>

The draw is the one in section 13 of task3_gan_sneha.ipynb, adapted to the team repo, where each member has their own root:
seed 42, the (member, direction) groups in sorted order, quotas 8 / 8 / 7 / 7, then one shuffle. Each group's candidates are
the stems that member actually translated (their pred_A2B / pred_B2A folders), because the two members translated different
photos. Each sample is saved as audit_XX.jpg: the input on the left, the translation on the right.

audit_key.json says which member, direction and image each sample is. Commit it, but don't open it until both rating
sheets are committed. The script refuses to redraw if the key already exists, so the set stays fixed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PRED = {"ayush": REPO / "ayush" / "task3_gan" / "ayush" / "outputs",      # the submitted epoch-100 outputs
        "sneha": REPO / "sneha" / "task3_gan" / "sneha" / "outputs"}      # the submitted Colab step A outputs
N, SEED = 30, 42

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=REPO / "sneha" / "task3_gan" / "data",
                help="folder holding the course's monet_jpg/ and photo_jpg/")
args = ap.parse_args()
src_dir = {"A2B": args.data / "monet_jpg", "B2A": args.data / "photo_jpg"}

if (HERE / "audit_key.json").exists():
    raise SystemExit("audit_key.json already exists: the audit set is fixed and is not redrawn")

groups = {(m, d): sorted(p.stem for p in (PRED[m] / f"pred_{d}").glob("*.jpg")) for m in sorted(PRED) for d in ("A2B", "B2A")}
for (m, d), stems in groups.items():
    assert len(stems) == 300, f"{m} pred_{d}: {len(stems)} images, expected 300"
    missing = [s for s in stems if not (src_dir[d] / f"{s}.jpg").exists()]
    assert not missing, f"{len(missing)} inputs of {m} pred_{d} not found in {src_dir[d]}"

rng = np.random.default_rng(SEED)
keys = sorted(groups)
quota = {k: N // len(keys) + (1 if i < N % len(keys) else 0) for i, k in enumerate(keys)}    # 8, 8, 7, 7
chosen = [(k[0], k[1], groups[k][i]) for k in keys for i in sorted(rng.choice(len(groups[k]), quota[k], replace=False))]
order = rng.permutation(len(chosen))

key = {}
for n, i in enumerate(order, 1):
    m, d, s = chosen[i]
    canvas = Image.new("RGB", (512, 256))
    canvas.paste(Image.open(src_dir[d] / f"{s}.jpg").convert("RGB").resize((256, 256)), (0, 0))
    canvas.paste(Image.open(PRED[m] / f"pred_{d}" / f"{s}.jpg").convert("RGB").resize((256, 256)), (256, 0))
    canvas.save(HERE / f"audit_{n:02d}.jpg", quality=95)
    key[f"audit_{n:02d}"] = {"member": m, "direction": d, "stem": s}

(HERE / "audit_key.json").write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8")
(HERE / "ratings_template.csv").write_text("sample,style,content,artifacts\n" + "".join(f"{k},,,\n" for k in key),
                                           encoding="utf-8")
print(f"built {len(key)} audit samples in {HERE.relative_to(REPO)}/ (quotas {list(quota.values())}); key written, not shown")
