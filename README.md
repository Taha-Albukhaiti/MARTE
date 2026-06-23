# MARTE

**Autor: Taha M. A. Al-Bukhaiti**

**Multi-Agent Red-Teaming Experiment** — Masterarbeit, Informatik M.Sc., Technische Hochschule Lübeck.

Untersucht, ob ein RAG-gestütztes Langzeitgedächtnis (LTM) die Angriffsleistung bei automatisierten Red-Teaming-Angriffen auf LLM-Tool-Agenten verbessert, verglichen mit reinem Kurzzeitgedächtnis (STM).

---

## Setup

Python ≥ 3.11 erforderlich.

```bash
git clone https://github.com/tahaalbukhaiti/MARTE.git
cd MARTE
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,llm,rag]"   # dev=Tests, llm=OpenAI-Client, rag=ChromaDB
cp .env.example .env              # API-Keys und Modellnamen eintragen
```

Wichtigste `.env`-Variablen:

```bash
LLM_BASE_URL=
ATTACK_LLM_MODEL=       # Planner + Mutator + Judge
TARGET_LLM_MODEL=       # Zielagent
```

Installation prüfen (kein API-Key nötig):

```bash
python main.py --mode local --runs-per-condition 1 --max-budget 3
```

---

## Experimente ausführen

```bash
# Smoke-Test ohne API-Aufrufe
python main.py --mode local --runs-per-condition 1 --max-budget 3

# Volles Experiment (alle 8 Conditions)
python main.py --mode api --runs-per-condition 30 --max-budget 50 --db-path results/events.db

# Nur bestimmte Conditions (Tokens: all, stm_only, rag_ltm, sql, file, moderate, enterprise)
python main.py --conditions sql_moderate_stm_only sql_moderate_rag_ltm

# Nach Unterbrechung fortsetzen
python main.py --mode api --runs-per-condition 30 --max-budget 50 \
    --db-path results/events.db \
    --resume-experiment-id exp_2026-06-14_04-53-49-234519
```

**Conditions:**

| Name | Szenario | Guardrail | Memory |
|---|---|---|---|
| `sql_moderate_stm_only` | sql | moderate | stm_only |
| `sql_moderate_rag_ltm` | sql | moderate | rag_ltm |
| `sql_enterprise_stm_only` | sql | enterprise | stm_only |
| `sql_enterprise_rag_ltm` | sql | enterprise | rag_ltm |
| `file_moderate_stm_only` | file | moderate | stm_only |
| `file_moderate_rag_ltm` | file | moderate | rag_ltm |
| `file_enterprise_stm_only` | file | enterprise | stm_only |
| `file_enterprise_rag_ltm` | file | enterprise | rag_ltm |

**Alle CLI-Optionen:**

| Option | Standard | Beschreibung |
|---|---|---|
| `--mode` | `local` | `local` = Stubs ohne API, `api` = echte LLM-Backends |
| `--runs-per-condition` | `30` | Runs pro Condition |
| `--max-budget` | `50` | Max. Attack-Steps pro Run |
| `--db-path` | `results/events.db` | SQLite-Pfad für Ergebnisse |
| `--attack-model` | aus `.env` | Modell für Planner + Mutator |
| `--target-model` | aus `.env` | Modell für Zielagent |
| `--judge-model` | wie attack-model | Modell für Judge |
| `--conditions` | `all` | Welche Conditions ausführen |
| `--resume-experiment-id` | — | Experiment-ID zum Fortsetzen |
| `--log-level` | — | `DEBUG`, `INFO` oder `WARNING` |

---

## Auswertung

MARTE hat zwei Auswertungspfade mit unterschiedlichen Rollen:

- `python -m marte.evaluation`: generische Projekt-Evaluation für eine einzelne EventStore-Datenbank. Gut für Smoke-Checks, laufende Experimente und maschinenlesbare Standardreports.
- `scripts/build_thesis_evaluation.py`: thesis-spezifischer Artefakt-Builder für die finalen vier Datenquellen. Er erzeugt die Tabellen, Abbildungen, Audits und Provenance-Dateien, die direkt in die Masterarbeit eingehen.

Das Thesis-Skript ersetzt den Evaluation-Layer nicht. Der Evaluation-Layer bleibt die wiederverwendbare Library/CLI für Metriken, Statistik und Datenvalidierung; das Skript ist der finale, read-only Report-Builder für die abgegebenen Experimentdaten.

### Schnellauswertung (laufendes Experiment)

```bash
python -m marte.evaluation --db-path results/events.db

# Mit Report-Ausgabe
python -m marte.evaluation --db-path results/events.db --output-dir results/reports/
```

Ausgabe: ASR, Median Q2B, Fisher's Exact Test, Log-Rank (Q2B + T2B), Holm-korrigierte p-Werte für alle 4 Vergleichspaare.

---

### Thesis-Auswertung (finale Artefakte)

Das Skript `scripts/build_thesis_evaluation.py` liest die vier finalen Datenbanken, validiert die Experimentstruktur und erzeugt alle Thesis-Artefakte in einem Durchgang.

```bash
python scripts/build_thesis_evaluation.py \
    --output-dir exp-results/thesis-evaluation \
    --chroma-path runtime/chroma

# Abbruch bei Datenfehlern (für CI)
python scripts/build_thesis_evaluation.py \
    --output-dir exp-results/thesis-evaluation \
    --chroma-path runtime/chroma \
    --strict
```

**Eingabe:** Die vier Datenbanken und die Chroma-LTM-Collection sind in `DEFAULT_SOURCES` fest kodiert (Experiment-IDs und Pfade). Eine alternative Konfiguration kann mit `--source-config config.json` übergeben werden.

**Ausgabe in `--output-dir`:**

| Datei / Ordner | Inhalt |
|---|---|
| `validation_report.json` | Datenqualitätsprüfung: Errors, Warnings, Issues |
| `condition_metrics.json/.csv` | ASR, KM-Median, RMST, Breach-Wahrscheinlichkeit je Condition |
| `pairwise_comparisons.json/.csv` | Risikodifferenz, Fisher's exact, Log-Rank, Holm-p je Block |
| `retrieval_mechanism.json/.csv` | RAG-Trefferquoten und Ähnlichkeitswerte je Phase |
| `ltm_audit.json/.csv` | Chroma-Integritätsprüfung: erwartete vs. tatsächliche Dokument-IDs |
| `source_manifests.csv` | SHA-256, Modelle und Budget je Quelldatenbank |
| `provenance.json` | Vollständige Herkunftsdokumentation |
| `manifest.json` | Quellspezifikationen und Holm-Familienplanung |
| `tables/condition_metrics.tex` | LaTeX-Tabelle der 8 Bedingungen |
| `tables/pairwise_comparisons.tex` | LaTeX-Tabelle der 4 Vergleichspaare |
| `figures/asr_risk_difference_forest.pdf` | Forest-Plot: Risikodifferenz RAG − STM mit 95%-CI |
| `figures/q2b_km_panels.pdf` | 2×2 Kaplan-Meier-Überlebenskurven (Q2B) |
| `figures/run_order_q2b.pdf` | Q2B nach Seed-Reihenfolge (Drift-Check) |
| `figures/retrieval_hitrate.pdf` | Balkendiagramm der RAG-Trefferquoten |

**Validierungsprüfungen (Auswahl):**

- Exakt 30 gültige Stop-Events pro Condition, Seeds 1–30 ohne Duplikate
- Seed-Parität zwischen STM- und RAG-Condition (gepaarter Vergleich)
- Ausführungsreihenfolge: ungerade Seeds starten mit STM, gerade mit RAG
- Oracle-Konsistenz: `breach_stop` ↔ `oracle_verdict=True` nur im letzten Schritt
- Schritt-Sequenz: `step_t` 1..q2b für jeden Run
- Keine Same-Seed-Retrieval-Treffer (LTM-Isolation)
- Chroma-Rekonstruktion: erwartete Dokument-IDs stimmen mit tatsächlichen überein

---

### Ergebnisübersicht (finale Experimentdaten)

| Condition | ASR | KM-Median Q2B | RMST₅₀ | Zensiert |
|---|---|---|---|---|
| `sql_moderate_stm_only` | 0.900 | 9 | 16.4 | 3/30 |
| `sql_moderate_rag_ltm` | **1.000** | 8 | 10.2 | 0/30 |
| `sql_enterprise_stm_only` | 0.000 | — | 50.0 | 30/30 |
| `sql_enterprise_rag_ltm` | 0.000 | — | 50.0 | 30/30 |
| `file_moderate_stm_only` | 0.833 | 10 | 18.7 | 5/30 |
| `file_moderate_rag_ltm` | **1.000** | **3** | **6.7** | 0/30 |
| `file_enterprise_stm_only` | 0.233 | — | 44.2 | 23/30 |
| `file_enterprise_rag_ltm` | **0.967** | **10** | **15.7** | 1/30 |

RMST₅₀ = Restricted Mean Survival Time bei τ = 50 Schritten (erwartete Schritte bis Breach; niedriger = schneller).

**Paarweise statistische Tests (Holm-korrigiert, α = 0.05):**

| Block | RD (RAG−STM) | Fisher Holm-p | Q2B Log-Rank Holm-p | Signifikant? |
|---|---|---|---|---|
| `sql_moderate` | +0.10 | 0.475 | 0.098 | Nein |
| `sql_enterprise` | 0.00 | 1.000 | — | Nein (0 Breaches) |
| `file_moderate` | +0.17 | 0.157 | **0.001** | Q2B ja, ASR nein |
| `file_enterprise` | **+0.73** | **< 0.001** | **< 0.001** | Ja (beide) |

RD = Risikodifferenz; 95%-KI nach Newcombe. Q2B-Holm-Familie: geplant m = 4, effektiv m = 3 (sql\_enterprise: keine Ereignisse → kein Test möglich).

---

## Tests

```bash
pytest tests/ -v
ruff check src/ tests/
```

---

## Architektur

```
1 Experiment
└── 8 Conditions  (2 Szenarien × 2 Guardrail-Level × 2 Memory-Modi)
    └── 30 Runs pro Condition  (Seeds 1–30)
        └── ≤ 50 Steps pro Run
            └── 1 Step = Planner → Mutator → TargetExecutor → BreachOracle → Judge
```

```
src/marte/
├── experiment/     # ComparativeRunner, Ausführungsreihenfolge, Seed-Kontrolle
├── attack/         # RunOrchestrator, Planner, Mutator, TargetExecutor, BreachOracle, Judge
├── memory/         # SwitchingMemoryGateway, STMRingBuffer (k=5), RAGAdapter, NoOpStrategy
├── target/         # TargetAgent, GuardrailEngine, ToolBroker, SandboxBoundary, SQL-/FileTool
├── persistence/    # EventStore (SQLite, append-only)
├── evaluation/     # generische Evaluation: MetricsService, StatsService, Validator, CLI
└── cli/            # Backends, ComponentFactory, Conditions, RunAdapter
```

Vollständige Architekturbeschreibung: `docs/ARCHITECTURE.md`
