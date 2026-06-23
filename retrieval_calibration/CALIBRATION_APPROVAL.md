# Calibration approval

- Calibration version: `rag-thresholds-v1-2026-06-13`
- Review date: June 13, 2026
- Status: approved for the main experiment
- Positive threshold: `0.5676532241107325`
- Blocked threshold: `0.6303290619675757`
- Outcome metrics used for selection: no
- Independent second annotation: no

The review covered all 80 query-episode pairs in `rag_pairs_review.csv`.
Channel assignment, scenario and guardrail strata, relevance labels, and
annotation reasons were checked for semantic consistency. The hard negative
pairs were retained because they deliberately distinguish lexical similarity
from usefulness for the concrete protected target.

The thresholds are the exact macro-F1 optima in
`rag_threshold_report.json`. They are frozen before the main experiment.
Later ASR, Q2B, T2B, breach counts, or significance results must not be used
to modify them.
