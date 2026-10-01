# DATA 266 Lab 1 (team scaffold)

Ayush folders are ready under each task. Sneha adds her own `sneha/` member directories. Do not overwrite each other's trees.

## Run (lab GPU or local)

```bash
cd ayush
python -m pip install -r requirements.txt
# Point task3_gan/data/{monet_jpg,photo_jpg} at the unzipped Kaggle images first.

# Task 1: 100,000 train sequences and 10,000 fixed val sequences, 30 epochs
python task1_llm/ayush/src/train.py --config configs/ayush/base.yaml

# Task 2 is already trained. Do not rerun unless the machine has no checkpoints.

# Task 3: one smoke step, then the full run from scratch
python task3_gan/ayush/src/train.py --config configs/ayush/base.yaml --smoke
python task3_gan/ayush/src/train.py --config configs/ayush/base.yaml --all-photos --epochs 80 --decay-start 40
python task3_gan/ayush/src/inference.py
```

Then score with `team/Part3_Evaluation_Script.ipynb` against the full `photo_jpg` folder. Upload the script's positive FID and MiFID. Do not negate them.

Dataset zip (Google Drive, read access): PASTE_LINK

No absolute personal paths. Device falls back to CPU if CUDA is missing. Task 3 refuses to train without CUDA.
