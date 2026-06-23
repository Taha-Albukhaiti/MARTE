# MARTE RAG calibration

This directory contains the retrieval-only calibration artifacts used to set
the positive and blocked cosine-similarity thresholds before the main
experiment.

The dataset is intentionally independent of ASR, Q2B, T2B, breach counts, and
statistical significance. Each query-episode pair is labeled only by whether
the stored episode is concretely useful for the current retrieval situation.

Generate the scored dataset with the same embedding model as MARTE:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/build_rag_calibration_dataset.py \
  --output results/retrieval_calibration/rag_pairs.json
```

The generator also writes `rag_pairs_review.csv`. This table contains the
query, candidate episode, assigned relevance label, annotation reason, and
calculated similarity for every pair.

Select the thresholds:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/calibrate_rag_thresholds.py \
  --input results/retrieval_calibration/rag_pairs.json \
  --output results/retrieval_calibration/rag_threshold_report.json
```

The labels and their reasons are defined explicitly in
`scripts/build_rag_calibration_dataset.py`. Their semantic and structural
review was completed on June 13, 2026. The approved values and the documented
limitation that no independent second annotation was used are recorded in
`CALIBRATION_SUMMARY.md`. The resulting thresholds are frozen for the main
experiment.
