# Ayush Task 3 — epoch-100 numbers for the team report

**Locked submission.** Do not use epoch 200. Phase-2 (101–200) made local FID worse (87.47 vs 85.70).

## Kaggle / local FID

| | value |
|---|---|
| Local avg FID | **85.6997** |
| Local MiFID | **0.4101** |
| FID A2B (Monet→photo) | 84.9805 |
| FID B2A (photo→Monet) | 86.4188 |
| Kaggle board (Team 21) | **≈ −43.05** (same as this CSV; top ~5) |
| CSV | `outputs/submission.csv` and `outputs/submission_epoch100/` |

## Training cost (submission run, RTX 5080 Laptop 16 GB)

| | value | note |
|---|---|---|
| Cycle L1 (epoch-100 mean) | **0.0761** | from train log |
| Identity L1 (epoch-100 mean) | 0.0523 | |
| Peak GPU memory | **9.88 GB** | |
| Train time epochs 50→100 | 26 381 s (~7.3 h) | exact `done metrics` |
| Approx. total epochs 1→100 | **~51 200 s (~14.2 h)** | sum of log segments (1–45 wall + 45–50 + 50–100); not one continuous job |
| Throughput (50→100) | 24.4 img/s | batch 4 |
| Params | 28 298 120 | |
| NaNs | 0 | |
| Device | NVIDIA GeForce RTX 5080 Laptop GPU | |
| Config | ResNet-9, LSGAN, cycle 10, identity 5, Adam 2e-4, EMA 0.999, DiffAugment, bf16 | |

## Not computed (leave blank in report — do not invent)

KID, density, coverage, LPIPS, content cosine. No Ayush metrics script was run for these. Sneha’s `metrics_report.csv` pattern can be reused later if needed.

## Raw logs (in repo)

`ayush/task3_gan/ayush/logs_from_gpu/`

| log | what |
|---|---|
| `task3_cyclegan_20261001T234814Z.log` | epochs ~1–45 |
| `task3_cyclegan_20261002T230821Z.log` | resume ~45→50 |
| `task3_cyclegan_20261003T003637Z.log` | **submission segment** resume 50→100 (`done metrics` at end) |
| `task3_cyclegan_20261003T233635Z.log` | phase-2 101→200 (**not used**) |

## Human audit

- Ayush sheet: `sneha/task3_gan/audit/ratings_ayush.csv` (done)
- Sneha sheet: `ratings_sneha.csv` (done)
- Agreement outputs: `agreement.csv`, `scores.csv` (from `python sneha/task3_gan/audit/agreement.py`)
