# Ayush Task 3 — epoch-100 numbers for the team report

**Locked submission.** Do not use epoch 40 or epoch 200.

## Kaggle / local FID

| | value |
|---|---|
| Local avg FID | **85.6997** |
| Local MiFID | **0.4101** |
| FID A2B (Monet→photo) | 84.9805 |
| FID B2A (photo→Monet) | 86.4188 |
| Kaggle board (Team 21) | **≈ −43.05** (~top 5) |
| CSV | `outputs/submission.csv` (= `outputs/submission_epoch100/`) |
| Checkpoint | `task3_gan/ayush/checkpoints/cyclegan_epoch100.pt` |

## Training cost (RTX 5080 Laptop 16 GB)

| | value |
|---|---|
| Cycle L1 (epoch-100 mean) | **0.0761** |
| Identity L1 (epoch-100 mean) | 0.0523 |
| Peak GPU memory | **9.88 GB** |
| Approx. total epochs 1→100 | **~51 200 s (~14.2 h)** |
| Final segment 50→100 | 26 381 s |
| NaNs | 0 |

## Not computed (leave blank — do not invent)

KID, density, coverage, LPIPS, content cosine.

## Raw logs (in git)

`reproducibility/raw_logs/ayush/`

| log | what |
|---|---|
| `task3_cyclegan_20261001T234814Z.log` | epochs ~1–45 |
| `task3_cyclegan_20261002T230821Z.log` | ~45→50 |
| `task3_cyclegan_20261003T003637Z.log` | **50→100 submission** (`done metrics`) |
| `task1_chargpt_*.log` / `task2_yelp_*.log` | Tasks 1–2 |

## Human audit

`sneha/task3_gan/audit/ratings_ayush.csv` + `agreement.csv` / `scores.csv`.

## Checkpoints zip (for Drive)

Ayush uploads `~/Documents/ayush_lab1_checkpoints.zip` (tasks 1–3 weights + logs). Drive (anyone with the link): https://drive.google.com/drive/folders/1Ks5qQSWtfKzqk8HL5nCBkAjM9iBW3Jha?usp=sharing
