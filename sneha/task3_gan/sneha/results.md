# Task 3 — CycleGAN, Monet ↔ Photo (Sneha)

> Template. Fill each section from your own run: numbers from `metrics_report.csv`
> and `submission.csv`, images from `outputs/`, and the reasoning in your own words
> (brief §8; viva §4).

**Run evidence:** notebook `src/task3_gan_sneha.ipynb` (saved with outputs) · generators
`checkpoints/generators_best.pt` (link: ______, sha256 in the manifest) · manifest
`reproducibility/manifests/sneha_task3.json` · raw logs listed in `logs.md`.

## 1. What I built (brief 3.1)

| Setting | Value | Why (your reasoning, tied to a number from your run) |
|---|---|---|
| Generator | U-Net-256 | |
| Decoder dropout (`unet_dropout`) | | |
| Discriminator | 70×70 PatchGAN (30×30 output) | |
| Losses | LSGAN + 10 × cycle L1 + 5 × identity L1 | |
| Optimiser / schedule | Adam 2e-4 (0.5, 0.999); `epochs: auto` fitted to `train_until`: ___ epochs, ___ at constant lr, then linear decay (from `train_summary.json` → `schedule`) | |
| Image pool / batch / augmentation | 50 / 1 / resize 286, crop 256, flip | |
| EMA generator weights | | |
| DiffAugment policy | translation, cutout (no colour) | |
| Parameters (measured) | | |
| Hardware, training time, images/s, peak memory | | |

Data: all 300 Monets; the 6,138 photos in `photo_train_all_6138.txt` (every photo outside the three held-out lists). The 300 reference
photos the provided script scores against (`ref_photos_300.txt`) are never trained on.
Generators chosen by validation FID, each on its own direction (`select_per_direction`): Monet → photo from epoch ___, photo → Monet from epoch ___. Stopped at `train_until`: yes / no.

## 2. Training behaviour and stability (brief 3.1.5, 3.2.3)

Embed `outputs/loss_curves.png` and two `samples_epoch_*.png` grids. Discuss generator vs
discriminator balance, the cycle and identity terms, gradient norms, NaN count, and how
validation FID moved across epochs.

## 3. Visual quality and cycle consistency (brief 3.2.1, 3.2.2)

Embed `outputs/translations_first4.png`. Discuss style, content and artifacts in both
directions. Cycle-consistency evidence: `cycle_l1_a`, `cycle_l1_b`, and the reconstructed
column of the sample grids.

## 4. Metrics, both directions

| Metric | Monet → photo (A2B) | Photo → Monet (B2A) |
|---|---|---|
| FID (provided script) | | |
| MiFID (provided script) | | |
| KID mean ± std | | |
| Density / coverage | | |
| Cycle-reconstruction L1 | | |
| LPIPS (input vs translation) | | |
| Content cosine (input vs translation) | | |

Final cycle / identity loss · max gradient norm G / D · NaN count · human audit score and
kappa per axis (team, `task3_gan/audit/agreement.csv`).

## 5. Kaggle (brief 3.2.5)

Team `PairProgramming_Team_##` · submitted `submission.csv` (sha256 in the manifest) ·
public score ___ · private score ___ · rank ___ on ______ (date).

## 6. Comparison with Ayush's model, shortcomings, next steps (brief 3.2.4)
