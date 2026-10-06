# Task 2 — Yelp Polarity sentiment classification (Sneha)

**Run evidence:** notebook `src/task2_sentiment_sneha.ipynb` (saved with outputs) ·
checkpoints `checkpoints/*_best.pt` · manifest `reproducibility/manifests/sneha_task2.json` ·
raw logs listed in `logs.md` · settings `config_used.yaml` (sha256 in its first line).

## 1. Data and preprocessing (brief 2.1)

- Split: shared files in `task2_sentiment/data/splits/` (100K train / 20K val stratified,
  seed 42; full 38K official test). Rows dropped by cleaning, from the preprocessing log: train 4 reviews empty after
  cleaning (100,000 → 99,996), val 1 (20,000 → 19,999), test none (38,000). 0 invalid labels, 0 duplicate texts,
  0 train reviews also in the test set, 0 val reviews also in train.
- Label mapping, quoted verbatim from the dataset's `readme.txt`: "The Yelp reviews polarity dataset is constructed by
  considering stars 1 and 2 negative, and 3 and 4 positive." In the CSVs, label 1 = negative and 2 = positive
  (the code maps `y = label − 1`).
- Review length in words (whitespace split): train mean 134.4, median 98, 90th percentile 285, 99th 625, max 1,023.
  Classes: train 49,997 negative / 49,999 positive; val 9,999 / 10,000; test 19,000 / 19,000.
  Plots: `outputs/length_distribution.png`, `outputs/class_distribution.png`, `outputs/class_balance.png`.
- Cleaning steps: the stored `\n` and `\"` are unescaped; lowercased; apostrophes removed (`don't` → `dont`); every other
  non-alphanumeric character becomes a space. Why: the CSVs store line breaks as the two characters `\n`, which would otherwise become a
  token "n"; lowercasing merges "Great" and "great"; dropping the apostrophe keeps "don't" as one token ("dont") instead of "don" + "t".
  The cost, which the error review found, is that emoticons and capitals are lost and "won't" no longer contains "not".
- Stopwords: scikit-learn's `ENGLISH_STOP_WORDS` minus `keep_words` (no, not, nor, never, cannot, none, nobody, nothing,
  neither, nowhere, but, however, although, though, very, too, few, less, least, more, most, against). Why: standard lists remove negations
  and contrast words such as "not", "no" and "but", which flip or qualify a review, so I protected them. The list still removes some
  sentiment-bearing words I did not protect — back, again, still, well, enough, only, even — which matters in the error review.
- Stemming: NLTK Snowball (English); no lemmatisation. Why: the brief allows either; Snowball needs no part-of-speech tags and is
  deterministic, and it merges inflections ("loved", "loving", "loves" → "love") so the 30,000-word vocabulary covers more of the data.
- Vocabulary 30,002 ids (the 30,000 most frequent words + `<pad>` + `<unk>`); out-of-vocabulary rate 0.61% train,
  0.86% val, 0.87% test. `max_len` 256 truncates 1.74% of train, 1.49% of val and 1.61% of test reviews
  (without stopword removal and stemming: 12.7%, 12.6%, 12.6%).
- **Did stopword removal + stemming help?** From `ablation_report.csv` (meanpool, validation): with both, macro-F1
  **0.9212** (accuracy 0.9212); with neither, macro-F1 **0.9269** (accuracy 0.9269); best epoch 3 in both.
  So no: on validation it **hurt**, by 0.0057 macro-F1, even though without stopword removal 12.6% of reviews are truncated at 256 tokens
  instead of 1.6%. The words the stopword list removes carry sentiment here ("will be back", "still a favourite", "well done"), and stemming
  may also merge some forms whose sentiment differs (the ablation switches both off together, so it can't separate the two). The test numbers below use the main pipeline (stopwords removed, stemmed) as configured;
  the ablation suggests the plain pipeline would do better.

## 2. Models and embedding choices (brief 2.2.2)

| Model | Architecture | Embedding | Params (measured) | Why |
|---|---|---|---|---|
| meanpool (baseline) | Masked mean of the word embeddings → Linear 128→256, ReLU, dropout 0.3 → Linear 256→256, ReLU → Linear 256→1 | nn.Embedding(30002, 128), learned from scratch | 3,939,329 | A deliberately minimal baseline: an average of word embeddings ignores word order completely, so anything the other two models gain over it is what order or characters are worth. |
| bigru | One bidirectional GRU layer (hidden 128 per direction) over packed sequences → max-pool over time (padding masked) → dropout 0.3 → Linear 256→1 | nn.Embedding(30002, 128), learned from scratch | 4,038,657 | A recurrent model that reads words in order in both directions, so it can use context such as negation; max-pooling keeps the strongest feature at any position. About the same size as the baseline, so the comparison is about order, not capacity. |
| charcnn | Zhang et al. CharCNN, learned 16-d character embedding instead of one-hot: 6 convolutions of 256 filters (kernels 7, 7, 3, 3, 3, 3; max-pool 3 after the 1st, 2nd and 6th) on 1,014 characters → FC 8704→512→512→1, dropout 0.5 | nn.Embedding(70, 16), learned from scratch (69 characters + padding) | 5,996,641 | The model from the paper that introduced this dataset (Zhang, Zhao and LeCun 2015), and one that never sees words: it tests whether characters alone reach word-level accuracy. The brief asks for embeddings learned from scratch, so a learned 16-d character embedding replaces the paper's one-hot input. |

Training for all three: Adam, lr 1e-3, batch 64, up to 5 epochs, early stopping on
validation macro-F1 (patience 2). Hardware per model (`device_name`, `cpu_name`): NVIDIA GeForce RTX 4090, x86_64
(SJSU GPU lab, Docker on WSL2, torch 2.1.2, CUDA 12.1) for all three. Best epoch: meanpool 3, bigru 3, charcnn 4 (5 epochs run each).

## 3. Metrics (all, per model)

From `metrics_report.csv`, on the full 38,000-review test set. Brackets are 95% bootstrap CIs (1,000 resamples).

| Metric | meanpool | bigru | charcnn |
|---|---|---|---|
| Accuracy [95% CI] | 0.9211 [0.9184, 0.9240] | 0.9382 [0.9359, 0.9407] | 0.9045 [0.9015, 0.9073] |
| Precision / recall / F1 (macro) | 0.9217 / 0.9211 / 0.9211 | 0.9383 / 0.9382 / 0.9382 | 0.9049 / 0.9045 / 0.9045 |
| Precision / recall / F1 (micro) | 0.9211 / 0.9211 / 0.9211 | 0.9382 / 0.9382 / 0.9382 | 0.9045 / 0.9045 / 0.9045 |
| Precision / recall / F1 (weighted) | 0.9217 / 0.9211 / 0.9211 | 0.9383 / 0.9382 / 0.9382 | 0.9049 / 0.9045 / 0.9045 |
| Confusion matrix (TN, FP, FN, TP) | 17,854, 1,146, 1,853, 17,147 | 17,927, 1,073, 1,275, 17,725 | 16,876, 2,124, 1,505, 17,495 |
| ROC-AUC / PR-AUC | 0.9762 / 0.9770 | 0.9848 / 0.9853 | 0.9678 / 0.9685 |
| MCC [95% CI] | 0.8427 [0.8372, 0.8486] | 0.8765 [0.8718, 0.8815] | 0.8094 [0.8034, 0.8149] |
| Macro-F1 [95% CI] | 0.9211 [0.9183, 0.9240] | 0.9382 [0.9359, 0.9407] | 0.9045 [0.9014, 0.9073] |
| Brier / ECE | 0.0587 / 0.0124 | 0.0462 / 0.0127 | 0.0699 / 0.0055 |
| McNemar vs baseline (statistic, p) | — | 199.20, 3.1e-45 | 98.32, 3.6e-23 |
| Slice short (n = 9,051): macro-F1, error rate | 0.9152, 0.0819 | 0.9308, 0.0665 | 0.8983, 0.0970 |
| Slice long (n = 7,583): macro-F1, error rate | 0.9184, 0.0758 | 0.9354, 0.0605 | 0.8619, 0.1313 |
| Slice negation (n = 27,991): macro-F1, error rate | 0.9129, 0.0835 | 0.9350, 0.0628 | 0.8982, 0.0990 |
| Slice contrast (n = 22,221): macro-F1, error rate | 0.9110, 0.0877 | 0.9313, 0.0680 | 0.8922, 0.1071 |
| Params / train seconds / examples per sec / peak memory | 3,939,329 / 18.9 / 26,477 / 0.22 GB | 4,038,657 / 194.9 / 2,565 / 0.41 GB | 5,996,641 / 50.1 / 9,981 / 1.21 GB |

Slices: short = at most 50 tokens, long = at least 200 tokens.

- **Accuracy, macro-F1, MCC:** bigru is best on every one (accuracy 0.938, MCC 0.876), meanpool second, charcnn third. The CIs don't overlap,
  so the ranking holds on this test set.
- **Precision / recall:** macro, micro and weighted are almost identical because the test set is exactly balanced (19,000 / 19,000).
- **Confusion matrix:** meanpool and bigru make more false negatives than false positives (1,853 vs 1,146; 1,275 vs 1,073), so they lean
  towards "negative"; charcnn leans the other way (2,124 false positives vs 1,505 false negatives).
- **ROC-AUC / PR-AUC:** all above 0.96, so the ranking of reviews is good even where the 0.5 threshold misclassifies; bigru ranks best (0.985).
- **Brier / ECE:** all three are well calibrated (ECE ≤ 0.013); charcnn has the lowest ECE (0.0055) but the highest Brier, because it is
  calibrated but less accurate.
- **McNemar:** both experimental models differ from the baseline with p far below 0.001 — bigru better, charcnn worse.
- **Slices:** bigru is best on every slice. charcnn's long-review slice is its weakest by far (macro-F1 0.862, error rate 13.1%), because it
  reads only the first 1,014 characters and a long review can run past that. For bigru, the weakest slice is the short reviews (0.931).
- **Cost:** bigru took 194.9 s to train (2,565 examples/s), 10 times meanpool's 18.9 s, for +1.7 points of accuracy.

## 4. Comparison (brief 2.3)

- **Against my baseline:** bigru beats meanpool by 1.7 points of accuracy (0.938 vs 0.921) and 0.034 MCC; the 95% CIs don't overlap and
  McNemar's p is 3.1e-45, so word order helps, by a real but modest margin. charcnn is *worse* than the baseline (0.905, McNemar p 3.6e-23):
  at this data size (100K reviews, 5 epochs), reading characters does not reach word-level accuracy, and it suffers most on long reviews.
  The bag-of-words baseline already gets 92.1%, which says most of the signal in Yelp polarity is in which words appear, not their order.
- **Against Ayush's models** (`ayush/task2_sentiment/ayush/results.md`): his BiLSTM + attention reaches 0.943 accuracy (MCC 0.887), his
  BiLSTM max-pool 0.937 and TextCNN 0.935; my bigru (0.938) sits with his max-pool BiLSTM, which is architecturally the closest model. Two
  differences favour him: he kept all words (no stopword removal or stemming, which my ablation also found better) and expanded "can't"/"won't";
  and attention pooling, which can weigh the closing verdict of a mixed review, is the step that made his best model better.
- **Strengths:** all three models are calibrated (ECE ≤ 0.013) and trained in minutes; bigru is accurate on negation and contrast
  slices (macro-F1 0.935 / 0.931). **Weaknesses:** mixed reviews whose verdict comes last, and preprocessing that deletes cues (emoticons,
  capitals, some sentiment-bearing stopwords; see `failure_analysis.md`). **Limitations:** 100K of the 560K training reviews, 5 epochs,
  one seed per model, so differences of a few tenths of a point are within run-to-run noise.
- **What I would try next:** first the preprocessing fixes from the error review (expand contractions, keep emoticons, protect back/again/still)
  on the plain-token pipeline; then attention pooling in place of max-pooling for bigru, which tests the "verdict at the end" errors directly.
