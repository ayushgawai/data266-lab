# Task 1 results

Character GPT trained from scratch on TinyStories. No `nn.Transformer`, no `nn.MultiheadAttention`, no `scaled_dot_product_attention`.

## Setup

100,000 train stories and 10,000 val stories, seed 42, from `roneneldan/TinyStories`. Vocabulary is the 108 characters that appear in the train stories. Each epoch draws 20,000 windows of 256 characters, batch 64, for 30 epochs.

Architecture: learned token embeddings, learned positional embeddings, 4 pre-norm blocks, 4 heads, d_model 256, d_ff 1024, GELU, dropout 0.1, weight tying. AdamW 3e-4, betas 0.9 and 0.95, weight decay 0.1 on matrices only, 200-step warmup, cosine down to 3e-5, grad clip 1.0. 3,248,640 parameters.

Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition, torch 2.11.0+cu128. Peak allocated memory 1.91 GB. Wall time 181.5 s. About 845k train tokens/s and 1,355 generated tokens/s.

## Numbers

| metric | value |
|---|---|
| train cross-entropy (epoch 30) | 0.810 |
| val cross-entropy (epoch 30) | 0.758 |
| best val cross-entropy | 0.753 (epoch 29) |
| perplexity at best val | 2.12 |
| bits per character at best val | 1.09 |
| generalization gap at epoch 30 | -0.051 |
| top-1 next character | 0.763 |
| distinct-1 / 2 / 3 | 0.0056 / 0.047 / 0.164 |
| repeated 4-gram rate | 0.823 |
| grad norm max / median | 7.18 / 0.595 |
| grad spikes / NaNs | 0 / 0 |

Epoch 1 val loss was 2.14. The saved checkpoint is epoch 29 at 0.753. Epoch 30 val was 0.758, a small bounce (this is CSV `val_ce` / `val_ce_epoch30`, and it matches the log line `epoch=30 ... val_loss=0.7584`). Each val pass draws a fresh 2,000 windows, so the **0.7514** in the log’s final `done metrics` `val_ce` is one more draw on the best weights — not a better epoch and not a second training run. Prefer best val **0.753** plus epoch-30 val **0.758** in the report. The negative gap is the epoch train average (dropout on) against the val pass at the end of that epoch.

Distinct-1 looks tiny because this is a character model and the vocab is only 108 symbols. Over 20 samples of 500 new characters, a few dozen unique letters are a small fraction of the token count. The repeated 4-gram rate is high for the same reason: English repeats chunks like " the" and " and", and the samples also loop phrases. The readable failure cases are in `failure_analysis.md`.
