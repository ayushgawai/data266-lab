# Task 3 results

Unpaired CycleGAN, Monet and photos. Generator is ResNet-9 for 256. Discriminator is a 70x70 PatchGAN (30x30 map, no sigmoid). LSGAN. Cycle weight 10. Identity weight 5. InstanceNorm affine, no running stats. Adam 2e-4, betas 0.5 / 0.999. Image pool 50. Train aug: resize 286, random crop 256, flip. Inference: resize 256. EMA 0.999 used at inference. DiffAugment on discriminator inputs. bf16 + channels-last.

## Submission run (lab GPU, RTX 5080 Laptop 16 GB)

- Data: all 300 Monets, 6,438 train photos (`photo_train.txt`). FID reference photos and val holdout stay out of training.
- Batch 4, ~1,610 steps/epoch, 12 loader workers.
- 100 epochs. LR constant through epoch 40, then linear decay.
- Checkpoints every 10 epochs. Weights on disk (gitignored): `checkpoints/from_gpu/cyclegan_epoch100.pt`.

### Scores

| checkpoint | local avg FID | local MiFID | Kaggle board |
|---|---|---|---|
| epoch 45 | 87.84 | 0.413 | -44.13 (Team 21) |
| epoch 50 | 86.73 | 0.412 | — |
| **epoch 100 (submission)** | **85.70** | **0.410** | **−43.05 (Team 21, ~top 5)** |
| epoch 200 (not used) | 87.47 | 0.409 | worse locally; discarded |

Local FID is `(FID_A2B + FID_B2A) / 2` from `evaluate_local.py` (same procedure as `team/Part3_Evaluation_Script.ipynb`).

Report package for Sneha: `epoch100_report_metrics.md` (cycle L1, train time, peak mem, log paths; KID/LPIPS/etc. not run). Human audit: `sneha/task3_gan/audit/ratings_ayush.csv` + `agreement.csv`.

Submission artifacts:

- `outputs/submission.csv` (= epoch-100 lock; also under `outputs/submission_epoch100/`)
- `outputs/pred_A2B/` (300 Monet → photo)
- `outputs/pred_B2A/` (300 photo → Monet)
- `outputs/fid_directions.txt`
- `logs_from_gpu/` (raw train logs)

```
ID,FID,MiFID
1,85.69966467190102,0.4100760370492935
```

## Older baseline (not submitted)

40 epochs, batch 1, 2,000-photo subset. Local avg FID 98.24.
