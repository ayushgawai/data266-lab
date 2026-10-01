# Lab 1 handoff (2026-09-30)

For the next coding session. Ayush Gawai, SJSU DATA 266, repo `data266-9689`. Partner Sneha is doing her half separately. Do not write her folders, Cohen's kappa, the team comparison, or the combined Report.pdf until she is done.

Due Oct 6 2026 6pm. Late until Oct 12. Canvas wants one zip of Part 1, Part 2, Part 3, plus one combined Report.pdf with the GitHub link. That zip is not built yet.

## Do not

- Do not start Colab. Credits were low and the G4 session was stopped on purpose on 2026-09-30 about 16:30 PT. `colab sessions` was empty after that.
- Do not use a TPU. The lab machine tomorrow is an RTX 5090.
- Do not invent KID, LPIPS, density, coverage, or a Kaggle rank. Missing numbers stay missing.
- Do not submit a negative FID. The leaderboard displays a minus sign. The file from the eval script is positive.
- Do not commit `dataset.zip`, `Lab1/resources/`, `*.pt`, raw logs, or other homework zips. Ayush will zip the big files himself.
- Do not reopen HW5. It is pushed at tag `hw5` (`ac6ff39`).

## What a clone has, and what it does not

Code, configs, writeups, the official `Part3_Evaluation_Script.ipynb`, generated eval images, and metrics CSVs are in git. These are not:

- Monet and photo JPEGs. `task3_gan/data/monet_jpg`, `photo_jpg`, and `real_stats.npz` are symlinks into `Lab1/resources/kaggle/extract/`, which was never committed. On the lab machine those three links will be broken until the Kaggle images are unzipped there. `photo_subset_2000.txt` is in git.
- Weights. `*.pt` is gitignored. Local disk still has `task3_gan/ayush/checkpoints/cyclegan_epoch40.pt` (108 MB, FID 98.24), plus the Task 1 and Task 2 checkpoints. Bring those in the zip. GitHub would also refuse the CycleGAN file because it is over 100 MB.

## Where the three tasks stand

Lab seed is 42. Ayush only.

Task 1, character GPT, done. No `nn.Transformer` / `MultiheadAttention` / `scaled_dot_product_attention`. 4 layers, 4 heads, d_model 256, d_ff 1024, pre-norm, weight tying, block 256, batch 64, 30 epochs, 100k train / 10k val TinyStories. Logged epoch 30 train CE 0.8099, val CE 0.7584, gap -0.0515. Best checkpoint is epoch 29, val CE 0.7526. Vocab 108, 3.25M params, RTX PRO 6000, torch 2.11.0+cu128. Notebook to show is `task1_llm/ayush/run.ipynb`. `task1_llm/ayush/src/run.ipynb` is an old unexecuted smoke notebook.

Task 2, Yelp polarity `fancyzhx/yelp_polarity`, done, second run. Full official test is 38,000 rows. Plain tokens beat stopwords plus WordNet lemma on the one-epoch ablation (val loss 0.207 vs 0.257). Contractions: `can't` and `won't` are expanded before the generic `n't` replace. Test metrics:

| model | accuracy | macro F1 | MCC |
|---|---|---|---|
| TextCNN | 0.935 | 0.935 | 0.869 |
| BiLSTM max-pool | 0.937 | 0.937 | 0.874 |
| BiLSTM + attention | 0.943 | 0.943 | 0.887 |

McNemar: TextCNN vs BiLSTM p=0.062. TextCNN vs attention p about 2e-13. BiLSTM vs attention p about 4e-9. Attention is the best model. 20 errors are 5 confident false positives, 5 confident false negatives, 5 near-threshold, 5 long reviews, from `bilstm_attn_errors.jsonl`. The current writeup is `failure_analysis.md`. Ignore any older 0.932-era numbers if a notebook still shows them.

Task 3, CycleGAN, trained, score is behind the board. ResNet-9 generator for 256, 70x70 PatchGAN with the last wide layer stride 1 so the map is 30x30, LSGAN, cycle L1 weight 10, identity 0.5 times cycle, InstanceNorm affine and no running stats, Adam 2e-4 betas 0.5 and 0.999, pool 50, batch 1 in the original run. 300 Monets, and a seed-42 subset of 2,000 photos out of 7,038. 40 epochs, learning rate constant 25 then decay to about 0. No NaNs. Official scorer wrote `task3_gan/ayush/outputs/submission.csv`:

- Monet to photo FID 98.98, MiFID 0.425
- Photo to Monet FID 97.50, MiFID 0.410
- Average FID 98.24, MiFID 0.418

`pred_A2B` and `pred_B2A` each have 300 JPEGs in git. A2B used all 300 Monets, including training images. Savitha Vijayarangan confirmed that is required: no held-out Monet split, every `monet_jpg` image gets a file in `pred_A2B`, same stems, JPEG quality 95.

The generator still leaves brush texture on photos and a pink weave on landscapes. That is why FID is about 98 and not about 45.

## Leaderboard, 2026-09-30 about 11:00 PT

Signup: https://www.kaggle.com/t/74a04fe5289c43c78f121340b56a841d

Competition: https://www.kaggle.com/competitions/data-266-fall-2026-gan-image-style-transfer

Team name must be `PairProgramming_Team_##`. Five submissions a day for the whole team. File `submission.csv` with header `ID, FID, MiFID` and one row. ID is numeric. Values must match the eval script. Example from staff: `1, 43.456, 0.389`.

The board sorts a higher score first and was showing negatives:

| rank | team | score |
|---|---|---|
| 1 | PairProgramming_Team_38 | -44.41 |
| 2 | PairProgramming_Team_40 | -46.99 |
| 3 | PairProgramming_Team_16 | -48.86 |

Those magnitudes match FID, with a minus sign. Our 98.24 has not been uploaded. On that reading it would sit near -98, under all three. Upload the positive script file anyway.

## Lab machine: RTX 5090, not Colab

Code as of 1 Oct:

- Task 1 now trains 100,000 fixed-length sequences and validates on a fixed 10,000. The old random-window run does not match the brief. Retrain it. Architecture is still pre-norm 4/4/256.
- Task 2 stays as trained. Do not rerun.
- Task 3 trains on `photo_train.txt` (6,438 photos). `ref_photos_300.txt` is the first 300 sorted photos and is never trained on. `val_photos_300.txt` is a seed-42 holdout and is also left out. Monet batches are reshuffled when the loader ends. Discriminator inputs get translation and cutout. Generators keep an EMA at 0.999, and inference uses those weights. Gradient clip is 100. bf16, channels-last, and batch 8 on 32 GB stay. The generator is still ResNet-9.

The epoch-40 checkpoint was trained on the wrong photo set. Do not resume it. Train from scratch.

```bash
python task1_llm/ayush/src/train.py --config configs/ayush/base.yaml
python task3_gan/ayush/src/train.py --config configs/ayush/base.yaml --smoke
python task3_gan/ayush/src/train.py --config configs/ayush/base.yaml --all-photos --epochs 80 --decay-start 40
python task3_gan/ayush/src/inference.py
```

Run these from `ayush/`. Then open `team/Part3_Evaluation_Script.ipynb` on the full `photo_jpg`. Upload the positive CSV. `--batch-size` overrides the automatic choice. If 80 epochs finishes with time left, run 40 more with `--resume` on the newest `cyclegan_epoch*.pt` and `--epochs 40 --decay-start 0`.

Shorter note with the same Kaggle rules: `Lab1/kaggle_submission_note.md`.

## Staff posts, Sep 27

Part 3 CSV is unchanged: team name `PairProgramming_Team_##`, file `submission.csv`, columns `ID, FID, MiFID`, one data row, numbers from the eval script.

Oct 1 course note: both members must train all three parts. The demo may show the team's best model, and the two of you may split which task each of you leads. Kaggle still needs a submission from each person. Part 2 is Yelp, not IMDB. Datasets cannot go on GitHub: zip them, put them on Drive with read access, and put that link in the README. Smoke-test on a tiny subset before the GPU lab slot. The deadline will not move.

## Sneha plan v3, file `DATA266_Lab1_Plan_Sneha_v3.html`

Models are unchanged: post-norm GPT (6/8/256), averaged embeddings, BiGRU max-pool, character CNN, CycleGAN U-Net 256 with a 70x70 PatchGAN. Updated 1 Oct: she trains on every photo outside three held-out lists (`ref_photos_300`, `eval_photos_300`, `val_photos_300`), about 6,138 photos. Run length is `epochs: auto` until a `train_until` cutoff, half constant lr and half decay, cap 400. She wants EMA 0.999, DiffAugment on discriminator inputs, and a separate best checkpoint per direction chosen by validation FID. She says first place on 1 Oct was 41.49, and the board score is minus the mean FID. Those extras need Ayush's agreement before they become the shared recipe. Her U-Net is about 218 MB in fp32, so it will not fit in a normal git push.

## Kiro review of the whole lab, 2026-10-01, Claude Opus 5.5, effort max

Task 1 is a correct pre-norm GPT and stays different from Sneha. The open question is the 100K/10K unit. Ayush's plan and `base.yaml` treat those as stories. Sneha's v3 says staff, on 15 Sep 2026, defined them as fixed-length sequences. The code takes 100K stories and draws 20,000 random windows per epoch. Confirm the unit before any retrain. Validation windows are redrawn each pass.

Task 2 is ready. Yelp, all 38,000 test rows, contractions, masked pool and attention, McNemar. Attention test accuracy 0.943. Stopwords looked worse partly because the list drops "not" and "no". Say that in the report.

Task 3 architecture matches the ResNet-9 plan and is distinct from Sneha's U-Net. The 98.24 score is far from a 1 Oct leader near 41.5, and nothing has been submitted. Defects called out: `itertools.cycle` freezes the Monet augmentations after the first 300 steps, the photo reference in the writeup may be the subset's first 300 rather than the script's first 300 of all photos, gradient clip is 10 while the max generator norm was 397, resume does not restore Adam or the learning-rate schedule, `--all-photos` includes the reference photos, and inference keeps the newest checkpoint rather than the best FID. Keep bf16, channels-last, TF32, and prefetch. Adopt her held-out lists, EMA, DiffAugment, and a validation-FID checkpoint only if Ayush agrees, so the generator stays the only difference. README still has no Drive link and still points at the old Colab notebook. Both members still need their own Kaggle uploads.

## Earlier Kiro note, Task 3 speed only, same day

Verdict on the current Ayush trainer: no. bf16, channels-last, and batch 8 are faster per image. They are not the setup most likely to beat a FID near 41. Concrete gaps versus her v3: `--all-photos` also trains on the held-out reference photos, and the trainer has no EMA, no DiffAugment, and no validation-FID checkpoint pick. Resume restarts the learning rate at full 2e-4 and does not restore Adam state. Keep the speed settings (bf16, channels-last, TF32, prefetch). Do not treat the unrun 45-epoch command as a guaranteed top score.
