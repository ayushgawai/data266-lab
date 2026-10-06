# Clarifications for the team report (Ayush metrics vs raw logs)

For Sneha / TAs. Canonical writeups stay `task*/ayush/results.md` and the committed `metrics_report.csv` files.

## Task 1 — val CE 0.7584 (CSV) vs 0.7514 (log `done metrics`)

**Same run.** No second evaluation log.

| Number | Where | Meaning |
|---|---|---|
| **0.7584** | log `epoch=30 ... val_loss=0.7584` and CSV `val_ce` / `val_ce_epoch30` | Val CE at the **end of epoch 30** |
| **0.7526** | log `epoch=29 ... val_loss=0.7526` and CSV `best_val_ce` | Val CE when **best checkpoint** was saved |
| **0.7514** | log `done metrics` `val_ce` | **Extra** val pass on the best weights after training; each pass draws a fresh 2 000 windows, so it moves a little |

**Use in the report:** epoch-30 val **0.758** and best val **0.753** (as in `task1_llm/ayush/results.md`). Do not treat 0.7514 as a better model — it is noise from a second draw.

## Task 2 — CSV acc vs log `tested model=... acc=`

| Model | Log `tested` line | `metrics_report.csv` |
|---|---|---|
| TextCNN | 0.9323 | **0.9346** |
| BiLSTM | 0.9394 | **0.9368** |
| BiLSTM+attn | 0.9405 | **0.9433** |

**Use the CSV** (and `task2_sentiment/ayush/results.md`). That table is the full official test suite (F1, MCC, slices, CIs, McNemar inputs) and is what the writeups already cite. There is **no separate eval log** to push. The short `tested model=...` lines in `task2_yelp_20260930T065220Z.log` do not match the committed CSV; treat them as superseded / not the report source.

## Task 3 — manifest

Original auto-written epoch-100 manifest was overwritten on disk when phase-2 finished at epoch 200. No untouched copy found. Current `reproducibility/manifests/ayush_task3.json` was corrected by hand on Oct 6 to point at the **submitted** `cyclegan_epoch100.pt` (see its `notes` field).

## Task 3 — KID / density / LPIPS / content cosine / cycle L1

Previously marked “not computed” in the report. **Now computed** (Oct 6) with the same definitions as Sneha’s notebook; see `task3_gan/ayush/full_metrics_report.csv` and `epoch100_report_metrics.md`.

| | A2B | B2A |
|---|---|---|
| KID mean ± std | 0.0158 ± 0.0022 | 0.0056 ± 0.0014 |
| Density / coverage | 1.140 / 0.920 | 0.529 / 0.770 |
| Content cosine | 0.791 | 0.778 |
| LPIPS | 0.340 | 0.353 |
| Cycle L1 | 0.047 | 0.073 |

## Checkpoints + dataset Drive

Anyone-with-link folder: https://drive.google.com/drive/folders/1Ks5qQSWtfKzqk8HL5nCBkAjM9iBW3Jha?usp=sharing  
(`ayush_lab1_checkpoints.zip` + Task 3 `dataset.zip`)
