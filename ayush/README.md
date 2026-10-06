# DATA 266 Lab 1 (team scaffold)

Ayush folders are ready under each task. Sneha adds her own `sneha/` member directories. Do not overwrite each other's trees.

## Submitted (Ayush)

- Task 1 / 2: trained; checkpoints on disk (gitignored `*.pt`). Raw logs in `reproducibility/raw_logs/ayush/`.
- Task 3: **epoch-100** CycleGAN submitted (FID 85.70 / MiFID 0.410). Manifest → `task3_gan/ayush/checkpoints/cyclegan_epoch100.pt`.
- Checkpoints Drive zip (anyone-with-link): [ayush_lab1_checkpoints](https://drive.google.com/drive/folders/1Ks5qQSWtfKzqk8HL5nCBkAjM9iBW3Jha?usp=sharing)

## Run (lab GPU or local) — only if retraining

```bash
cd ayush
python -m pip install -r requirements.txt
# Point task3_gan/data/{monet_jpg,photo_jpg} at the unzipped Kaggle images first.

python task1_llm/ayush/src/train.py --config configs/ayush/base.yaml
# Task 2: already trained.
python task3_gan/ayush/src/train.py --config configs/ayush/base.yaml --all-photos --epochs 100 --decay-start 40
python task3_gan/ayush/src/inference.py
```

Score with `team/Part3_Evaluation_Script.ipynb`. Upload the script's positive FID and MiFID.

Dataset zip (Google Drive, read access): PASTE_LINK

No absolute personal paths. Device falls back to CPU if CUDA is missing. Task 3 refuses to train without CUDA.
