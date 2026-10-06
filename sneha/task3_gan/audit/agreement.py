"""Inter-rater agreement and scores for the Task 3 human audit. Run it only after both rating sheets are committed.

    python sneha/task3_gan/audit/agreement.py

Reads ratings_ayush.csv and ratings_sneha.csv (each a filled copy of ratings_template.csv: 1-5 per axis, 5 = best) and writes
  agreement.csv  quadratic-weighted Cohen's kappa, exact and within-1 agreement per axis (the same audit_agreement as the notebook)
  scores.csv     mean score per member and direction, both raters averaged; this uses audit_key.json, i.e. it unblinds the set
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
AXES = ["style", "content", "artifacts"]
RATERS = ["ayush", "sneha"]


def audit_agreement(r1: np.ndarray, r2: np.ndarray) -> dict:
    """Quadratic-weighted Cohen's kappa (ordinal 1-5 ratings), exact and within-1 agreement."""
    r1, r2 = np.asarray(r1), np.asarray(r2)
    k = 1.0 if np.array_equal(r1, r2) else float(cohen_kappa_score(r1, r2, weights="quadratic", labels=[1, 2, 3, 4, 5]))
    return {"kappa": k, "agree_exact": float((r1 == r2).mean()), "agree_within1": float((abs(r1 - r2) <= 1).mean())}


def load(rater: str, samples: list[str]) -> pd.DataFrame:
    df = pd.read_csv(HERE / f"ratings_{rater}.csv").set_index("sample")
    assert sorted(df.index) == samples, f"ratings_{rater}.csv must rate exactly the {len(samples)} audit samples"
    df = df[AXES]
    assert df.notna().all().all(), f"ratings_{rater}.csv has blank cells"
    assert df.isin([1, 2, 3, 4, 5]).all().all(), f"ratings_{rater}.csv: every rating must be a whole number 1-5"
    return df.astype(int).sort_index()


key = json.loads((HERE / "audit_key.json").read_text(encoding="utf-8"))
r = {name: load(name, sorted(key)) for name in RATERS}

agree = pd.DataFrame({axis: audit_agreement(r["ayush"][axis].to_numpy(), r["sneha"][axis].to_numpy()) for axis in AXES}).T
agree["mean_ayush"] = r["ayush"].mean()
agree["mean_sneha"] = r["sneha"].mean()
agree["mean_both"] = (r["ayush"] + r["sneha"]).mean() / 2
agree.index.name = "axis"
agree.round(4).to_csv(HERE / "agreement.csv")

both = (r["ayush"] + r["sneha"]) / 2
both["member"] = [key[s]["member"] for s in both.index]
both["direction"] = [key[s]["direction"] for s in both.index]
scores = both.groupby(["member", "direction"])[AXES].mean()
scores["n"] = both.groupby(["member", "direction"]).size()
overall = both.groupby("member")[AXES].mean().assign(direction="both", n=both.groupby("member").size())
scores = pd.concat([scores.reset_index(), overall.reset_index()]).sort_values(["member", "direction"])
scores["mean_3_axes"] = scores[AXES].mean(axis=1)
scores.round(3).to_csv(HERE / "scores.csv", index=False)

print(agree.round(3).to_string(), "\n")
print(scores.round(2).to_string(index=False))
