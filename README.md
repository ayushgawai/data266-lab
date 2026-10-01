# data266-lab

DATA 266 Lab 1, split so Ayush, Sneha, and the shared files stay apart.

| folder | what is in it |
|---|---|
| `ayush/` | runnable code, configs, results, checkpoints on disk |
| `sneha/` | Sneha's plans. Her code goes here when she adds it |
| `team/` | lab brief, official FID notebook, Kaggle notes |
| `resources/` | Kaggle images and downloads. Not in git |
| `dataset.zip` | the dataset archive. Not in git |

The photo folders under `ayush/task3_gan/data/` are links into `resources/`. Keep that folder next to `ayush/` when you unzip.

On the GPU machine, unzip this project, then follow `ayush/README.md`.

```bash
cd ayush
python -m pip install -r requirements.txt
```

GitHub has the code only. Weights (`*.pt`) and `resources/` stay in the full zip.
