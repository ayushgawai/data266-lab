# Task 3 results

Unpaired CycleGAN, Monet and photos. Generator is the 9-block ResNet used for 256 images. Discriminator is a 70x70 PatchGAN. It outputs a 30 by 30 score map and has no sigmoid. Least-squares loss. Cycle weight 10. Identity weight 5, which is 0.5 times the cycle weight. InstanceNorm with affine and no running stats. Adam 2e-4, betas 0.5 and 0.999. Batch size 1. Constant learning rate for 25 epochs, then linear decay over the last 15. Image pool of 50. Train augmentation is resize 286, random crop 256, horizontal flip. Inference is resize 256 only.

Both generators and both discriminators together are 28,298,120 parameters.

Data: all 300 Monet paintings, and 2,000 photos drawn with seed 42 from the 7,038 photos, listed in `task3_gan/data/photo_subset_2000.txt`. There was no shared subset file in the repo, so this seed is the one in `configs/ayush/base.yaml`.

## Training

40 epochs, 2,000 steps each, on an NVIDIA RTX PRO 6000 Blackwell. 3,583 seconds, about 45 images per second, peak memory 2.80 GB. No NaN losses. The largest generator gradient before clipping was 397. The clip is 10.

Last-epoch means: generator loss 3.18, cycle L1 0.152, identity L1 0.125. Discriminator losses stayed near 0.13 and 0.17, so the generators did not collapse the discriminators to zero. The curve is `outputs/loss_curves.png`. A sample strip from epoch 40 is `outputs/samples/epoch_040.png`: a Monet painting becomes a sunset photograph of the same building, and a photo becomes a brushy painting. The building and the tree line stay put.

## Official score

`outputs/submission.csv` was produced by the same Inception-v3 procedure as `Part3_Evaluation_Script.ipynb`, not typed in. Both directions use 300 images. Real photos are the first 300 names in sorted order from the seed-42 subset. Generated files keep the source stem, JPEG quality 95.

| direction | FID | MiFID |
|---|---|---|
| Monet to photo | 98.98 | 0.425 |
| Photo to Monet | 97.50 | 0.410 |
| average, the submission row | 98.24 | 0.418 |

Cycle L1 on the 300 eval images, without augmentation, is 0.081 from Monet back to Monet and 0.093 from photo back to photo.

KID, density, coverage, LPIPS, and a Kaggle rank are not in this folder. The leaderboard score has not been submitted. The two shortcomings that show up in the pictures are in `failure_analysis.md`.
