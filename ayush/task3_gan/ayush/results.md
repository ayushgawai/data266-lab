# Task 3 results

Unpaired CycleGAN, Monet and photos. Generator is ResNet-9 for 256. Discriminator is a 70x70 PatchGAN (30x30 map, no sigmoid). LSGAN. Cycle weight 10. Identity weight 5. InstanceNorm affine, no running stats. Adam 2e-4, betas 0.5 / 0.999. Image pool 50. Train aug: resize 286, random crop 256, flip. Inference: resize 256. EMA 0.999 used at inference. DiffAugment on discriminator inputs. bf16 + channels-last.

## Current run (lab GPU, RTX 5080 Laptop 16 GB)

- Data: all 300 Monets, 6,438 train photos (`photo_train.txt`). FID reference photos and val holdout stay out of training.
- Batch 4, ~1,610 steps/epoch, 12 loader workers.
- Schedule: LR constant through epoch 40, then linear decay. Target 100 epochs.
- Checkpoints every 10 epochs + `cyclegan_latest.pt`.

### Scores so far

| checkpoint | local avg FID | local MiFID | Kaggle board |
|---|---|---|---|
| epoch 45 | 87.84 | 0.413 | **-44.13** (rank 5, Team 21) |
| epoch 50 | 86.73 | 0.412 | pending submit |

Local FID is `(FID_A2B + FID_B2A) / 2` from `evaluate_local.py` (same procedure as `team/Part3_Evaluation_Script.ipynb`). Kaggle shows a negated board score from its own scoring of the CSV.

Submission file: `outputs/submission.csv` (epoch 50 numbers as of this writeup).

## Older baseline (do not resume)

40 epochs, batch 1, 2,000-photo subset only. Local avg FID 98.24. Weights removed after the full-photo run started.
