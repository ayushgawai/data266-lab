# Ayush raw training logs

All member logs live here (checklist). Checkpoints stay on Drive / local disk (`*.pt` gitignored; ~411 MB CycleGAN).

| File | Task | Role |
|---|---|---|
| `task1_chargpt_20260916T022433Z.log` | 1 | early / short run |
| `task1_chargpt_20260916T023604Z.log` | 1 | early / short run |
| `task1_chargpt_20260930T063834Z.log` | 1 | **final CharGPT train** (best ckpt epoch 29) |
| `task2_yelp_20260930T065220Z.log` | 2 | **final Yelp train** (TextCNN / BiLSTM / attn) |
| `task3_cyclegan_20261001T234326Z.log` | 3 | smoke |
| `task3_cyclegan_20261001T234814Z.log` | 3 | epochs ~1–45 |
| `task3_cyclegan_20261002T230821Z.log` | 3 | resume ~45→50 |
| `task3_cyclegan_20261003T003637Z.log` | 3 | **submitted** resume 50→100 (`done metrics`, FID 85.70) |
| `task3_cyclegan_20261003T233635Z.log` | 3 | phase-2 101→200 (**not submitted**; FID got worse) |

Manifests: `../manifests/ayush_task{1,2,3}.json` — Task 3 points at `cyclegan_epoch100.pt`.

Report numbers: `../../task3_gan/ayush/epoch100_report_metrics.md` and `SNEHA_AGENT_HANDOFF.md`.
