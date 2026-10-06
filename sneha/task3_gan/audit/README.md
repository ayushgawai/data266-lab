# Task 3 — blinded human audit (team)

Brief 3.2.6: 30 fixed samples, rated for style, content and artifacts by two raters, with inter-rater agreement.

| File | What it is |
|---|---|
| `build_audit.py` | Drew the set once (seed 42, 8 / 8 / 7 / 7 across member × direction, then shuffled). Refuses to redraw. |
| `audit_01.jpg` … `audit_30.jpg` | Input on the left, translation on the right. |
| `audit_key.json` | Which member, direction and image each sample is. **Don't open it until both rating sheets are committed.** |
| `rate.html` | Rating page with the rubric: open it in a browser, rate, press *Download CSV*. |
| `ratings_template.csv` | The same sheet as a plain CSV, if you'd rather fill it in by hand. |
| `ratings_ayush.csv`, `ratings_sneha.csv` | Each rater's sheet (1–5, 5 = best), filled in independently. |
| `agreement.py` → `agreement.csv`, `scores.csv` | Quadratic-weighted Cohen's kappa, exact and within-1 agreement per axis; mean scores per member and direction. |

Steps: each of us rates on our own, without discussing → both commit their `ratings_<name>.csv` → run
`python sneha/task3_gan/audit/agreement.py` → copy the results into the report.
