# Lab 1 Part 3 Kaggle note

Saved 2026-09-30 from the Canvas update. Do this when we submit. Do not change the numbers by hand.

## Signup

https://www.kaggle.com/t/74a04fe5289c43c78f121340b56a841d

Competition: https://www.kaggle.com/competitions/data-266-fall-2026-gan-image-style-transfer

Team name on Kaggle must be PairProgramming_Team_##. Up to 5 submissions a day for the whole team.

## File

Name: submission.csv

3 columns: ID, FID, MiFID

2 rows: header, then one result row

Example from staff:

ID, FID, MiFID
1, 43.456, 0.389

ID is numeric. FID and MiFID must be the numbers from Part3_Evaluation_Script.ipynb. Ours are already in ayush/task3_gan/ayush/outputs/submission.csv:

ID,FID,MiFID
1,98.23681993386393,0.4177033603191376

Staff asked teams to start submitting and to keep improving the model. Multiple submissions are allowed.

## Leaderboard snapshot, 2026-09-30 about 11:00 PT

The board sorts a higher score first. Only three teams had submitted.

| rank | team | score | entries | last |
|---|---|---|---|---|
| 1 | PairProgramming_Team_38 | -44.4076 | 3 | 16h |
| 2 | PairProgramming_Team_40 | -46.9872 | 1 | 12h |
| 3 | PairProgramming_Team_16 | -48.8553 | 2 | 2d |

We are not on the board. Our FID is 98.24. The posted scores are the same size as a FID, with a minus sign, so those teams are around 44 to 49. On that reading our 98 would sit under all three, near -98. Submit the positive script values, not a negated copy.

## Host answer, 2026-09-30

Question from Akash: can pred_A2B use all 300 Monet paintings, including ones used in training?

Savitha Vijayarangan: yes. There is no held-out split. Every image in monet_jpg gets a translated output in pred_A2B, so the leaderboard stays comparable.

Our run already did that. Training used all 300 Monets, and pred_A2B has 300 files. This post does not change the score.

## Next training, stopped 2026-09-30 16:30 PT

Colab G4 is stopped. Credits were running out. Do not start it again.

A continuation on all 7,038 photos reached epoch 44 of 45 once, cycle L1 about 0.106, and the VM disappeared before those weights were copied. They are gone. The file still on disk is `task3_gan/ayush/checkpoints/cyclegan_epoch40.pt` (FID 98.24).

Tomorrow, GPU lab, RTX 5090. Same generator and losses. The trainer now picks batch 8 on a 32 GB card, bf16, and channels-last, so one epoch is a few hundred steps instead of 7,038.

```
python task3_gan/ayush/src/train.py --config configs/ayush/base.yaml --all-photos --resume task3_gan/ayush/checkpoints/cyclegan_epoch40.pt --epochs 45 --decay-start 30
```

Photos and Monets have to be on that machine under `task3_gan/data/` (the Kaggle zip). Then run `inference.py` and `Part3_Evaluation_Script.ipynb`. Submit the positive FID and MiFID. Target is under 44.
