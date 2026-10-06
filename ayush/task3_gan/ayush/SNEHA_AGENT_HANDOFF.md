# Handoff for Sneha’s agent (Task 3)

Ayush side is locked for the report. Read this first.

## Done

- Kaggle CSV submitted from **epoch 100** only (`ayush/task3_gan/ayush/outputs/submission.csv`). Board ≈ **−43.05**, Team 21, ~top 5.
- Human audit: `sneha/task3_gan/audit/ratings_ayush.csv` committed. Sneha’s `ratings_sneha.csv` already present. `agreement.py` outputs: `agreement.csv`, `scores.csv`.
- Epoch-100 cost / cycle metrics + raw logs: see `ayush/task3_gan/ayush/epoch100_report_metrics.md` and `ayush/task3_gan/ayush/logs_from_gpu/`.

## Do not

- Do not use epoch 200 weights or `outputs/submission_epoch200/` (if present). Local FID got worse.
- Do not invent KID, density, coverage, LPIPS, or content cosine for Ayush — they were **not** computed. Leave blank or compute yourself with the same pipeline as `sneha/metrics_report.csv`.
- Do not retrain Ayush Task 3 for the report.

## Still team-side (not Ayush solo)

- Paste audit agreement + Ayush metrics into the combined Report.pdf.
- Canvas zip / final report packaging if not done.
