# Task 2 — 20-error review (Sneha)

> Template. The notebook (section 14) writes the 20 candidates, picked by the fixed rules, to
> `outputs/error_review_candidates.md`: 5 confident false positives, 5 confident false
> negatives, 5 near-threshold errors and 5 from the worst slice. Copy them here, then fill in
> **an error type and one testable fix for each, yourself** (brief 2.2.4).

Model reviewed (best validation macro-F1): · checkpoint: · worst slice:
Near-threshold errors inside [0.45, 0.55]: (if fewer than 5, the rest are the errors closest to 0.5)

Suggested error types: sarcasm / irony · negation · mixed sentiment (contrast) · label noise ·
sentiment carried by a rare or truncated word · review cut off at max_len · domain-specific phrase.

## Confident false positives

| # | Test row | p(positive) | Review (excerpt) | Error type | Testable fix |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

## Confident false negatives

| # | Test row | p(positive) | Review (excerpt) | Error type | Testable fix |
|---|---|---|---|---|---|
| 6 | | | | | |
| 7 | | | | | |
| 8 | | | | | |
| 9 | | | | | |
| 10 | | | | | |

## Near-threshold errors

| # | Test row | p(positive) | Review (excerpt) | Error type | Testable fix |
|---|---|---|---|---|---|
| 11 | | | | | |
| 12 | | | | | |
| 13 | | | | | |
| 14 | | | | | |
| 15 | | | | | |

## Worst-slice failures (slice: )

| # | Test row | p(positive) | Review (excerpt) | Error type | Testable fix |
|---|---|---|---|---|---|
| 16 | | | | | |
| 17 | | | | | |
| 18 | | | | | |
| 19 | | | | | |
| 20 | | | | | |

## Patterns across the 20

