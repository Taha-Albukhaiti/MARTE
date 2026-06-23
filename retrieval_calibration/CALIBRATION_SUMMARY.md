# Calibration summary

## Purpose

Calibration determines the minimum cosine similarity required for an LTM
episode to be included in the Planner or Mutator context. It does not optimize
attack success.

## Dataset

The dataset contains 80 query-episode pairs.

- 40 pairs belong to the positive-memory channel.
- 40 pairs belong to the blocked-pattern channel.
- Each channel contains 20 relevant and 20 irrelevant pairs.
- Every scenario and guardrail combination contains five relevant and five
  irrelevant pairs per channel.
- Labels are based only on semantic retrieval relevance.
- ASR, Q2B, T2B, breach counts, and p-values are not used.

The similarities were calculated with `all-MiniLM-L6-v2`, which is also the
embedding model used by MARTE.

## Approved thresholds

The selection rule maximizes macro-F1 across the four scenario and guardrail
strata. The higher threshold is selected on ties.

| Channel | Frozen threshold | Macro-F1 |
|---|---:|---:|
| Positive memory | 0.5676532241107325 | 0.743590 |
| Blocked patterns | 0.6303290619675757 | 0.833117 |

The exact values are frozen in calibration version
`rag-thresholds-v1-2026-06-13`:

```text
MARTE_RAG_POSITIVE_MIN_SIMILARITY=0.5676532241107325
MARTE_RAG_BLOCKED_MIN_SIMILARITY=0.6303290619675757
```

The positive threshold is not rounded to `0.568`, because that rounding
changes one boundary classification and lowers the calibration score.

## Interpretation

The blocked-pattern channel separates the labeled cases more reliably than
the positive channel. The lower positive score is caused mainly by hard
negatives with the same strategy and action but a different protected target.
For example, an episode about writing a public file can be lexically similar
to a query about writing a protected file.

The labels, annotation reasons, channel balance, and stratum balance in
`rag_pairs_review.csv` were reviewed on June 13, 2026. No label was changed.
The review confirmed that the deliberately difficult irrelevant pairs test
target-specific relevance rather than only lexical similarity. The thresholds
are approved for the main experiment and must not be changed using experiment
outcomes.

No independent second annotation was performed. This is recorded as a
limitation rather than being presented as inter-annotator agreement.
