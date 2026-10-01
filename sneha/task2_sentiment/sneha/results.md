# Task 2 — Yelp Polarity sentiment classification (Sneha)

> Template. Fill each section from your own run: numbers from `metrics_report.csv` and
> `ablation_report.csv`, plots from `outputs/`, and the reasoning in your own words
> (brief §8; viva §4).

**Run evidence:** notebook `src/task2_sentiment_sneha.ipynb` (saved with outputs) ·
checkpoints `checkpoints/*_best.pt` · manifest `reproducibility/manifests/sneha_task2.json` ·
raw logs listed in `logs.md` · settings `config_used.yaml` (sha256 in its first line).

## 1. Data and preprocessing (brief 2.1)

- Split: shared files in `task2_sentiment/data/splits/` (100K train / 20K val stratified,
  seed 42; full 38K official test). Rows dropped by cleaning, from the preprocessing log:
- Label mapping, quoted verbatim from the dataset's `readme.txt`:
- Review length distribution, class distribution, class balance (embed the three plots):
- Cleaning steps and why: lowercasing, punctuation, the stored `\n`, apostrophes (`don't` → `dont`):
- Stopword list and the words kept (`keep_words`), and why:
- Stemming vs lemmatisation, and why:
- Vocabulary size, `max_len` 256, and how many reviews it truncates:
- **Did stopword removal + stemming help?** Cite `ablation_report.csv` (validation macro-F1, both runs):

## 2. Models and embedding choices (brief 2.2.2)

| Model | Architecture | Embedding | Params (measured) | Why |
|---|---|---|---|---|
| meanpool (baseline) | | | | |
| bigru | | | | |
| charcnn | Zhang et al. CharCNN, learned 16-d character embedding instead of one-hot | nn.Embedding(70, 16), learned from scratch | | |

Training for all three: Adam, lr 1e-3, batch 64, up to 5 epochs, early stopping on
validation macro-F1 (patience 2). Hardware per model (`device_name`, `cpu_name`):

## 3. Metrics (all, per model)

Copy the three rows from `metrics_report.csv`; one sentence of interpretation per metric.

| Metric | meanpool | bigru | charcnn |
|---|---|---|---|
| Accuracy [95% CI] | | | |
| Precision / recall / F1 (macro, micro, weighted) | | | |
| Confusion matrix (TN, FP, FN, TP) | | | |
| ROC-AUC / PR-AUC | | | |
| MCC [95% CI] | | | |
| Macro-F1 [95% CI] | | | |
| Brier / ECE | | | |
| McNemar vs baseline (statistic, p) | — | | |
| Slices: short / long / negation / contrast (n, macro-F1, error rate) | | | |
| Params / train seconds / examples per sec / peak memory | | | |

## 4. Comparison (brief 2.3)

- My experimental models against my baseline. What do the CIs and the McNemar p-values license me to claim?
- Against Ayush's models (team table):
- Strengths, weaknesses, limitations:
- What I would try next:
