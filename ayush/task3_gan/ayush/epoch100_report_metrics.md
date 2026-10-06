# Ayush Task 3 — epoch-100 numbers for the team report

**Locked submission.** Do not use epoch 40 or epoch 200.

## Kaggle / local FID

| | value |
|---|---|
| Local avg FID | **85.6997** |
| Local MiFID | **0.4101** |
| FID A2B / B2A | 84.9805 / 86.4188 |
| Kaggle board (Team 21) | **≈ −43.05** (~top 5) |
| Checkpoint | `task3_gan/ayush/checkpoints/cyclegan_epoch100.pt` |

## Extra metrics (computed Oct 6 on RTX 5080; same defs as Sneha notebook)

Source: `full_metrics_report.csv` from `compute_extra_metrics.py` (Inception features; polynomial KID 100×100; density/coverage k=5; LPIPS AlexNet; EMA generators for cycle/LPIPS).

| Metric | A2B (Monet→photo) | B2A (photo→Monet) |
|---|---|---|
| KID mean ± std | **0.0158 ± 0.0022** | **0.0056 ± 0.0014** |
| Density / coverage | **1.140 / 0.920** | **0.529 / 0.770** |
| Content cosine | **0.791** | **0.778** |
| LPIPS | **0.340** | **0.353** |
| Cycle-recon L1 | **0.047** (A) | **0.073** (B) |

## Training cost (RTX 5080 Laptop 16 GB)

| | value |
|---|---|
| Final-epoch cycle / identity L1 (train log) | 0.0761 / 0.0523 |
| Peak GPU memory | 9.88 GB |
| Approx. total epochs 1→100 | ~51 200 s (~14.2 h) |

## Raw logs

`reproducibility/raw_logs/ayush/` — see `README.md` there.

## Checkpoints + dataset Drive

https://drive.google.com/drive/folders/1Ks5qQSWtfKzqk8HL5nCBkAjM9iBW3Jha?usp=sharing
