# Task 1 — Character-level GPT on TinyStories (Sneha)

> Template. Fill each section from your own run: numbers from `metrics_report.csv`,
> curves from `outputs/`, and the reasoning in your own words (brief §8; viva §4).

**Run evidence:** checkpoint `checkpoints/gpt_best.pt` · manifest
`reproducibility/manifests/sneha_task1.json` · raw logs listed in `logs.md` ·
config `config_used.yaml` (sha256 in its first line).

## 1. What I built

Decoder-only GPT, written from scratch (section 6 of `src/task1_llm_sneha.ipynb`): learned token and positional
embeddings, N post-norm Transformer blocks (masked multi-head self-attention and a
feed-forward network, each with a residual connection and a LayerNorm), and a linear
LM head over the character vocabulary.

| Setting | Value | Why (your reasoning, tied to a number from your run) |
|---|---|---|
| Layers / heads / d_model / d_ff | | |
| Norm position | post | |
| Context length | | |
| Activation | | |
| Dropout | | |
| Final LayerNorm (`final_ln`) | | |
| Weight tying | | |
| Optimizer / betas / weight decay | | |
| Peak LR / warmup / schedule | | |
| Gradient clipping | | |
| Batch size / epochs (1 epoch = 100K sequences) | | |
| Parameter count (measured) | | |

## 2. Data

- Split: 100K train / 10K val fixed-length sequences of 129 chars.
  Why validation is the *next* 10K sequences rather than a random 10K:
- Slice offset (`split_offset_bytes`), seed 4242:
- Vocabulary size, and how many validation characters map to `<unk>`:
- How `<|endoftext|>` is handled, and why:

## 3. Training behaviour

Embed `outputs/loss_curve.png` and `outputs/train_steps.png`, then discuss:
convergence, the effect of warmup on post-norm, gradient-norm behaviour, spikes, NaNs,
and where validation CE bottoms out (`best_epoch`) compared with the final epoch.

## 4. Metrics

Copy the row from `metrics_report.csv`. For each metric, give one sentence on what
it says about *this* model.

| Metric | Value | Interpretation |
|---|---|---|
| Train CE / Val CE (nats) | | |
| Perplexity / bits per char | | |
| Generalization gap (`gen_gap`, `gen_gap_eval`) | | |
| Top-1 next-char accuracy | | |
| Distinct-1/2/3, repeated 4-gram rate | | |
| Grad norm max / median, spikes, NaN count | | |
| Train tokens/s, generation tokens/s | | |
| Peak memory, total training time | | |
| Hardware | | |

## 5. Comparison with Ayush's model

One structural difference, and the measured consequence of it.

## 6. Limitations and what I would try next
