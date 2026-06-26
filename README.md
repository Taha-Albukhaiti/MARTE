# MARTE

**Author: Taha M. A. Al-Bukhaiti**

**Multi-Agent Red-Teaming Experiment** — Master's thesis, M.Sc. Computer Science, Technische Hochschule Lübeck.

Investigates whether a RAG-based long-term memory (LTM) improves attack performance in automated red-teaming attacks against LLM tool agents, compared to short-term memory (STM) only.

---

## Setup

Python ≥ 3.11 required.

```bash
git clone https://github.com/tahaalbukhaiti/MARTE.git
cd MARTE
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,llm,rag]"   # dev=tests, llm=OpenAI client, rag=ChromaDB
cp .env.example .env              # fill in API keys and model names
```

Key `.env` variables:

```bash
LLM_BASE_URL=https://models.mylab.th-luebeck.dev/v1
ATTACK_LLM_MODEL=glm-4.7        # Planner + Mutator + Judge
TARGET_LLM_MODEL=gpt-oss-120b   # Target agent
```

Verify installation (no API key required):

```bash
python main.py --mode local --runs-per-condition 1 --max-budget 3
```

---

## Running Experiments

```bash
# Smoke test without API calls
python main.py --mode local --runs-per-condition 1 --max-budget 3

# Full experiment (all 8 conditions)
python main.py --mode api --runs-per-condition 30 --max-budget 50 --db-path results/events.db

# Specific conditions only (tokens: all, stm_only, rag_ltm, sql, file, moderate, enterprise)
python main.py --conditions sql_moderate_stm_only sql_moderate_rag_ltm

# Resume after interruption
python main.py --mode api --runs-per-condition 30 --max-budget 50 \
    --db-path results/events.db \
    --resume-experiment-id exp_2026-06-14_04-53-49-234519
```

**Conditions:**

| Name | Scenario | Guardrail | Memory |
|---|---|---|---|
| `sql_moderate_stm_only` | sql | moderate | stm_only |
| `sql_moderate_rag_ltm` | sql | moderate | rag_ltm |
| `sql_enterprise_stm_only` | sql | enterprise | stm_only |
| `sql_enterprise_rag_ltm` | sql | enterprise | rag_ltm |
| `file_moderate_stm_only` | file | moderate | stm_only |
| `file_moderate_rag_ltm` | file | moderate | rag_ltm |
| `file_enterprise_stm_only` | file | enterprise | stm_only |
| `file_enterprise_rag_ltm` | file | enterprise | rag_ltm |

**All CLI options:**

| Option | Default | Description |
|---|---|---|
| `--mode` | `local` | `local` = stubs without API, `api` = real LLM backends |
| `--runs-per-condition` | `30` | Runs per condition |
| `--max-budget` | `50` | Max attack steps per run |
| `--db-path` | `results/events.db` | SQLite path for results |
| `--attack-model` | from `.env` | Model for Planner + Mutator |
| `--target-model` | from `.env` | Model for target agent |
| `--judge-model` | same as attack-model | Model for Judge |
| `--conditions` | `all` | Which conditions to run |
| `--resume-experiment-id` | — | Experiment ID to resume |
| `--log-level` | — | `DEBUG`, `INFO`, or `WARNING` |

---

## Evaluation

MARTE has two evaluation paths with distinct roles:

- `python -m marte.evaluation`: generic project evaluation for a single EventStore database. Useful for smoke checks, ongoing experiments, and machine-readable standard reports.
- `scripts/build_thesis_evaluation.py`: thesis-specific artifact builder for the four final data sources. Generates the tables, figures, audits, and provenance files that feed directly into the master's thesis.

The thesis script does not replace the evaluation layer. The evaluation layer remains the reusable library/CLI for metrics, statistics, and data validation; the script is the final, read-only report builder for the submitted experiment data.

### Quick Evaluation (ongoing experiment)

```bash
python -m marte.evaluation --db-path results/events.db

# With report output
python -m marte.evaluation --db-path results/events.db --output-dir results/reports/
```

Output: ASR, median Q2B, Fisher's exact test, log-rank (Q2B + T2B), Holm-corrected p-values for all 4 comparison pairs.

---

### Thesis Evaluation (final artifacts)

The script `scripts/build_thesis_evaluation.py` reads the four final databases, validates the experiment structure, and generates all thesis artifacts in a single pass.

```bash
python scripts/build_thesis_evaluation.py \
    --output-dir exp-results/thesis-evaluation \
    --chroma-path runtime/chroma

# Abort on data errors (for CI)
python scripts/build_thesis_evaluation.py \
    --output-dir exp-results/thesis-evaluation \
    --chroma-path runtime/chroma \
    --strict
```

**Input:** The four databases and the Chroma LTM collection are hard-coded in `DEFAULT_SOURCES` (experiment IDs and paths). An alternative configuration can be passed with `--source-config config.json`.

**Output in `--output-dir`:**

| File / Directory | Contents |
|---|---|
| `validation_report.json` | Data quality checks: errors, warnings, issues |
| `condition_metrics.json/.csv` | ASR, KM median, RMST, breach probability per condition |
| `pairwise_comparisons.json/.csv` | Risk difference, Fisher's exact, log-rank, Holm-p per block |
| `retrieval_mechanism.json/.csv` | RAG hit rates and similarity scores per phase |
| `ltm_audit.json/.csv` | Chroma integrity check: expected vs. actual document IDs |
| `source_manifests.csv` | SHA-256, models, and budget per source database |
| `provenance.json` | Complete provenance documentation |
| `manifest.json` | Source specifications and Holm family planning |
| `tables/condition_metrics.tex` | LaTeX table of the 8 conditions |
| `tables/pairwise_comparisons.tex` | LaTeX table of the 4 comparison pairs |
| `figures/asr_risk_difference_forest.pdf` | Forest plot: risk difference RAG − STM with 95% CI |
| `figures/q2b_km_panels.pdf` | 2×2 Kaplan-Meier survival curves (Q2B) |
| `figures/run_order_q2b.pdf` | Q2B by seed order (drift check) |
| `figures/retrieval_hitrate.pdf` | Bar chart of RAG hit rates |

**Validation checks (selection):**

- Exactly 30 valid stop events per condition, seeds 1–30 without duplicates
- Seed parity between STM and RAG condition (paired comparison)
- Execution order: odd seeds start with STM, even seeds with RAG
- Oracle consistency: `breach_stop` ↔ `oracle_verdict=True` only in the last step
- Step sequence: `step_t` 1..q2b for each run
- No same-seed retrieval hits (LTM isolation)
- Chroma reconstruction: expected document IDs match actual ones

---

### Results Summary (final experiment data)

| Condition | ASR | KM Median Q2B | RMST₅₀ | Censored |
|---|---|---|---|---|
| `sql_moderate_stm_only` | 0.967 | 11 | — | 1/30 |
| `sql_moderate_rag_ltm` | **1.000** | **6** | — | 0/30 |
| `sql_enterprise_stm_only` | 0.000 | — | — | 30/30 |
| `sql_enterprise_rag_ltm` | 0.000 | — | — | 30/30 |
| `file_moderate_stm_only` | 0.633 | 43 | — | 11/30 |
| `file_moderate_rag_ltm` | **0.933** | **6** | — | 2/30 |
| `file_enterprise_stm_only` | 0.400 | — | — | 18/30 |
| `file_enterprise_rag_ltm` | **0.967** | **6** | — | 1/30 |

RMST₅₀ = Restricted Mean Survival Time at τ = 50 steps (expected steps to breach; lower = faster). KM median "—" means the survival curve does not cross the 0.5 threshold within the budget.

**Pairwise statistical tests (Holm-corrected, α = 0.05):**

| Block | RD (RAG−STM) | Fisher Holm-p | Q2B Log-Rank Holm-p | Significant? |
|---|---|---|---|---|
| `sql_moderate` | +0.033 | 1.000 | **0.001** | Q2B yes, ASR no |
| `sql_enterprise` | 0.000 | 1.000 | — | No (0 breaches in both arms) |
| `file_moderate` | **+0.300** | **0.031** | **< 0.001** | Yes (both) |
| `file_enterprise` | **+0.567** | **< 0.001** | **< 0.001** | Yes (both) |

RD = risk difference; 95% CI by Newcombe. Q2B Holm family: planned m = 4, effective m = 3 (sql\_enterprise: no events → no test possible).

---

## Tests

```bash
pytest tests/ -v
ruff check src/ tests/
```

---

## Architecture

```
1 Experiment
└── 8 Conditions  (2 scenarios × 2 guardrail levels × 2 memory modes)
    └── 30 Runs per condition  (seeds 1–30)
        └── ≤ 50 Steps per run
            └── 1 Step = Planner → Mutator → TargetExecutor → BreachOracle → Judge
```

```
src/marte/
├── experiment/     # ComparativeRunner, execution order, seed control
├── attack/         # RunOrchestrator, Planner, Mutator, TargetExecutor, BreachOracle, Judge
├── memory/         # SwitchingMemoryGateway, STMRingBuffer (k=5), RAGAdapter, NoOpStrategy
├── target/         # TargetAgent, GuardrailEngine, ToolBroker, SandboxBoundary, SQL-/FileTool
├── persistence/    # EventStore (SQLite, append-only)
├── evaluation/     # generic evaluation: MetricsService, StatsService, Validator, CLI
└── cli/            # backends, ComponentFactory, Conditions, RunAdapter
```

Full architecture description: `docs/ARCHITECTURE.md`
