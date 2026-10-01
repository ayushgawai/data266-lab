# Task 2 results

Yelp polarity, three models trained from scratch. Embeddings are random and learned. No GloVe and no pretrained language model.

This is the rerun after the review. The test set is the full official 38,000 rows. "can't" maps to "can not" and "won't" maps to "will not".

## Data

`fancyzhx/yelp_polarity`: 560,000 train reviews, 38,000 test reviews. Split, seed 42, stratified, from the official train split only: 100,000 train and 20,000 validation. `test_idx.npy` is the official test order. Empty and duplicate cleaned reviews are dropped from train and val only (5 empty, 18 duplicates). Every test row is scored.

A one-epoch TextCNN check compared plain tokens with sklearn stopwords plus WordNet lemmatization. Plain val loss 0.207, stopword plus lemma val loss 0.257. The stopword list removes "not" and "no", which throws away the words that flip a review. Full training uses plain tokens. Vocabulary is the 30,000 most common train types plus `<pad>` and `<unk>`, max length 256.

## Models

Adam 1e-3, batch 64, up to 5 epochs, early stop when val loss stalls for 2 epochs. Saved weights are the best val-loss epoch.

| model | test acc | f1 macro | ROC AUC | MCC | Brier | ECE | 95% acc CI | params | train sec |
|---|---|---|---|---|---|---|---|---|---|
| TextCNN | 0.935 | 0.935 | 0.982 | 0.869 | 0.050 | 0.011 | 0.932 to 0.937 | 3,994,201 | 14 |
| BiLSTM max-pool | 0.937 | 0.937 | 0.985 | 0.874 | 0.047 | 0.008 | 0.935 to 0.939 | 4,104,449 | 94 |
| BiLSTM + attention | 0.943 | 0.943 | 0.986 | 0.887 | 0.043 | 0.006 | 0.941 to 0.946 | 4,137,473 | 95 |

Macro-F1 and MCC 95% intervals are in `metrics_report.csv`. For the attention model, macro-F1 is 0.941 to 0.946 and MCC is 0.882 to 0.891.

McNemar on the same 38,000 rows: TextCNN vs max-pool BiLSTM p = 0.062, so that step is not a clear win. TextCNN vs attention p about 2e-13, and max-pool vs attention p about 4e-9. Attention is the model that moves the number.

## Slices, attention model

| slice | n | macro F1 | error rate |
|---|---|---|---|
| short, under 50 tokens | 8,785 | 0.941 | 0.056 |
| long, over 200 tokens | 7,911 | 0.921 | 0.074 |
| negation | 28,186 | 0.940 | 0.058 |
| contrast | 22,525 | 0.935 | 0.065 |

Long reviews are still the weak slice. The 20-error read is in `failure_analysis.md`. Confusion matrices and reliability plots are in `outputs/`.

Hardware: NVIDIA RTX PRO 6000 Blackwell. Peak memory about 0.54 GB.
