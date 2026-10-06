# data266-lab — DATA 266 Lab 1, PairProgramming_Team_21

Ayush Sunil Gawai and Sneha Tumkur Narendra. LLM pretraining (Task 1), Yelp sentiment classification (Task 2) and CycleGAN
Monet ↔ photo (Task 3). Combined report: **[`report/DATA266_Lab1_Report_Team_21.pdf`](report/DATA266_Lab1_Report_Team_21.pdf)**.

## Layout

Each member's work has its own root, and inside it the brief's layout: `task1_llm/`, `task2_sentiment/`, `task3_gan/`, each with
the member's folder (`src/`, `checkpoints/`, `outputs/`, `metrics_report.csv`, `results.md`, `failure_analysis.md`), plus
`reproducibility/manifests/` and `reproducibility/raw_logs/`. We kept the two roots apart so neither of us could overwrite the
other's runs, configs or logs.

| Folder | What is in it |
|---|---|
| `ayush/` | Ayush's code, configs, results, raw logs and manifests · instructions in [`ayush/README.md`](ayush/README.md) |
| `sneha/` | Sneha's notebooks, configs, results, raw logs and manifests · instructions in [`sneha/README.md`](sneha/README.md) |
| `sneha/task3_gan/audit/` | The team's blinded human audit for Task 3 (30 samples, both raters, Cohen's kappa) |
| `team/` | The lab brief, the provided Task 3 evaluation script, Kaggle notes |
| `report/` | The combined PDF report and its Markdown source |

## Reproduce a run (one command)

Smoke test of Sneha's Task 1 notebook, end to end on a tiny model (about a minute on a laptop CPU; it downloads the 22 MB
TinyStories validation file):

```bash
cd sneha && python -m pip install -r requirements.txt && python smoke_test.py
```

`python smoke_test.py --task 2`, `--task 3` or `--task all` smoke-test the other notebooks. Ayush's runs: `cd ayush`, then the
commands in `ayush/README.md`.

## Not in git (too large): data and weights

- **Datasets:** zipped on Google Drive, read access: **[DATA266_Lab1_Team21_datasets](https://drive.google.com/drive/folders/1nr_CEwNylk7HZYPPP_ufqi1lwnONS_f9?usp=sharing)**. TinyStories V2 (Task 1), Yelp Polarity (Task 2) and the course's
  Monet / photo `dataset.zip` (Task 3). Sources and where to unzip them: [`sneha/README.md`](sneha/README.md#datasets).
- **Checkpoints:** Sneha's, all three tasks, [Google Drive](https://drive.google.com/drive/folders/1O6CWrI-I-WKmtVCej4znQIg0UbCRvxUB?usp=sharing);
  Ayush's, all three tasks, [Google Drive](https://drive.google.com/drive/folders/1Ks5qQSWtfKzqk8HL5nCBkAjM9iBW3Jha?usp=sharing).
- Raw logs record the working directory that each run printed when it started; all code reads its paths from config files.
