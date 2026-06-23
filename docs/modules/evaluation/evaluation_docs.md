# Evaluation Layer — Implementierungsdokumentation

## 1. Ziel und Abgrenzung

Der Evaluation Layer berechnet und testet die statistische Auswertung der MARTE-Experimente.
Er ist verantwortlich für:

- metrische Aggregation aus persistenten Experimentdaten (`MetricsService`)
- statistische Gruppenvergleiche zwischen Conditions (`StatsService`)
- Validitätsprüfung des gespeicherten Experiments vor der Auswertung (`EvaluationValidator`)

**Systemgrenze (bewusst):**
- Evaluation führt keine Angriffe aus.
- Evaluation ruft keine Target- oder Attack-Agenten auf.
- Evaluation liest ausschließlich aus dem `EventStore` (SQLite).

Warum diese Grenze: Metrik- und Statistiklogik muss reproduzierbar und unabhängig vom Laufzeitverhalten der Agenten bleiben.

### 1.1 Abgrenzung zum Thesis-Artefakt-Builder

`src/marte/evaluation` ist die generische Projekt-Evaluation. Sie wertet eine
EventStore-Datenbank aus und eignet sich für Smoke-Checks, laufende Experimente
und Standardreports.

`scripts/build_thesis_evaluation.py` ist dagegen der finale, thesis-spezifische
Artefakt-Builder. Er liest die vier eingefrorenen Ergebnisquellen, führt
zusätzliche Audits aus und erzeugt die CSV-, JSON-, LaTeX- und PDF-Artefakte
für die Masterarbeit. Dieses Skript ist absichtlich außerhalb von `src/marte`,
damit die finale Berichtserzeugung nicht Teil des experimentellen
Laufzeitpakets ist.

Beide Pfade bleiben deshalb sinnvoll: Der Evaluation Layer stellt die
wiederverwendbare Metrik-, Statistik- und Validierungsbasis bereit; das
Thesis-Skript orchestriert die finale mehrquellige Berichtserzeugung.

---

## 2. Rolle im MARTE-Gesamtsystem

Der Datenfluss für die Auswertung ist:

1. `ComparativeRunner` schreibt `step_events`, `stop_events` und `run_events` in SQLite.
2. `EvaluationValidator` prüft Condition-Matrix, Runzahlen, Seeds, Pairing, offene/abgebrochene Runs und StopEvent-Invarianten.
3. `MetricsService` liest `stop_events` je `experiment_id` und `condition`.
4. `StatsService` berechnet auf Basis der Metrikserien:
   - Fisher’s Exact Test für ASR
   - Log-Rank-Test für Q2B
   - Log-Rank-Test für Time-to-Breach
5. Die CLI ergänzt Holm-korrigierte p-Werte je Metrikfamilie.

Das entspricht der in `docs/ARCHITECTURE.md` beschriebenen Trennung:
Persistenz zuerst, danach Metrikaggregation, danach Inferenzstatistik.

---

## 3. Designprinzipien und Begründung

### 3.1 Persistenzgetriebene Auswertung
- Auswertung arbeitet auf `EventStore`-Daten statt auf In-Memory-Objekten.

Warum: Ergebnisse bleiben über Prozessgrenzen und Neustarts stabil reproduzierbar.

### 3.2 Strikte Filterung per `experiment_id`
- Alle Auswertungen filtern zuerst nach `experiment_id`.

Warum: Mehrere Experimente können in derselben SQLite-Datei koexistieren, ohne sich zu vermischen.

### 3.3 Zensierungsbewusste Survival-Logik
- Q2B und T2B behalten `censored_flag`.
- Log-Rank nutzt `event_observed = not censored_flag`.

Warum: `budget_exhausted` ist rechtszensiert, nicht automatisch ein „Misserfolg“ im Survival-Sinn.

### 3.4 Defensiver Fallback bei unzureichender Datenlage
- Für Survival-Tests gilt Mindestanzahl je Arm (`_MIN_OBSERVATIONS = 2`).
- Bei zu wenigen Beobachtungen werden p-Werte als `NaN` zurückgegeben.
- Wenn beide Gruppen vollständig zensiert sind, wird ebenfalls `NaN` zurückgegeben.

Warum: Ein numerischer Wert ohne ausreichende Daten wäre methodisch irreführend.

### 3.5 Vorabvalidierung statt stiller Reports
- Die CLI führt vor der Report-Erzeugung `EvaluationValidator.check_all()` aus.
- Ohne `--strict` werden Verstöße als Warnungen ausgegeben und im JSON-Report dokumentiert.
- Mit `--strict` bricht die Evaluation bei `error`-Verstößen mit Exit-Code 1 ab.

Warum: Ein unvollständiger Pilotlauf oder ein Crash darf nicht wie ein vollständiges 8×30-Experiment aussehen.

---

## 4. Komponentenübersicht

| Komponente | Datei | Aufgabe |
|---|---|---|
| `MetricsService` | `metrics.py` | ASR, Q2B-, T2B-Verteilungen pro Condition |
| `StatsService`, `ComparisonResult` | `stats.py` | Fisher + Log-Rank, Rückgabe als strukturiertes Ergebnis |
| `EvaluationValidator`, `Violation` | `evaluation_validator.py` | Integritätsprüfung vor der Statistik |
| CLI | `__main__.py` | Text-/JSON-Report, `--strict`, `--expected-runs`, `--max-budget`, Holm-Korrektur |
| Thesis-Artefakt-Builder | `../../scripts/build_thesis_evaluation.py` | Finale vierquellige Auswertung, Tabellen, Figuren, LTM-Audit und Provenance |
| `EventStore` (Dependency) | `../persistence/event_store.py` | Lesen der `stop_events`, `run_events`, optional `step_events` |
| Schemas (Dependency) | `../persistence/schemas.py` | Typisierte Events (`StopEvent`, `RunEvent`, `StepEvent`) |

---

## 5. Datenverträge und Invarianten

## 5.1 `StopEvent` (Input für Evaluation)
Aus `schemas.py`:
- `experiment_id`, `run_id`, `condition`, `stop_reason`, `q2b`, `censored_flag`, `timestamp_ms`, `seed`

Invarianten für korrekte Evaluation:
- `stop_reason == "breach_stop"` impliziert `censored_flag=False` und `q2b <= B`
- `stop_reason == "budget_exhausted"` impliziert `censored_flag=True` und `q2b == B`
- Die persistierten Schrittindizes reichen lückenlos von `1` bis `q2b`.
- Bei `breach_stop` enthält genau der letzte Schritt das erste positive Oracle-Urteil.
- Bei `budget_exhausted` liegt kein positives Oracle-Urteil vor.
- Die Stop-Dauer liegt nicht vor dem Zeitstempel des letzten Schritts.
- `stop_reason == "error"` ist kein gültiger Auswertungsrun
- pro `run_id` darf es genau ein `StopEvent` geben
- pro Condition müssen Seeds `1..N` ohne Lücken/Duplikate vorhanden sein
- `timestamp_ms` ist integer pro Run

## 5.2 `ComparisonResult` (Output der Statistik)
Aus `stats.py`:
- `asr_p_value`
- `asr_odds_ratio`
- `q2b_p_value`
- `t2b_p_value`

Wichtige Konvention:
- `NaN` signalisiert „nicht ausreichend beobachtbare Daten für valide Schätzung/Testung“.

---

## 6. `MetricsService` im Detail (`metrics.py`)

## 6.1 `compute_asr(experiment_id, condition) -> float | None`
Implementierung:
- lädt `stop_events` für das Experiment
- filtert auf Condition
- zählt `stop_reason == "breach_stop"`
- berechnet `breaches / total`
- Rückgabe `None` bei leerer Condition

Formel:
`ASR = (# breach_stop runs) / (# runs in condition)`

Warum `None` bei leerer Condition: Keine Daten sind nicht dasselbe wie 0% Angriffserfolg. Die CLI zeigt dafür `No data` im Textreport und `"n/a"` im JSON.

## 6.2 `get_q2b_distribution(...) -> list[tuple[int, bool]]`
Rückgabeformat:
- `(q2b, censored_flag)` pro Run

Begründung:
- Zensierung bleibt explizit erhalten, damit Survival-Methoden korrekt angewendet werden können.

## 6.3 `get_t2b_distribution(...) -> list[tuple[int, bool]]`
Rückgabeformat:
- `(timestamp_ms, censored_flag)` pro Run

Begründung:
- auch für Zeitmetrik wird dieselbe Zensierungssemantik benötigt.

---

## 7. `StatsService` im Detail (`stats.py`)

## 7.1 ASR-Test: Fisher’s Exact
Methode: `_compare_asr(...)`

Schritte:
1. Events beider Conditions laden
2. 2x2-Kontingenztafel bilden:
   - Breach vs. Nicht-Breach je Condition
3. `scipy.stats.fisher_exact` ausführen
4. `(odds_ratio, p_value)` zurückgeben

Fallback:
- Wenn ein Arm leer ist: `(NaN, NaN)`

Warum Fisher:
- robust bei kleinen Stichproben und diskreten Häufigkeiten.

## 7.2 Survival-Tests: Log-Rank für Q2B und T2B
Methode: `_compare_survival(..., metric="q2b"|"t2b")`

Schritte:
1. passende Verteilung aus `MetricsService` holen
2. Mindestanzahl prüfen (`_MIN_OBSERVATIONS`)
3. Dauervektoren bilden (`durations_*`)
4. Eventindikator bilden (`event_observed = not censored_flag`)
5. prüfen, ob mindestens ein Breach-Event in einer der beiden Gruppen beobachtet wurde
6. `lifelines.statistics.logrank_test(...)` ausführen
7. p-Wert zurückgeben

Fallback:
- Bei zu wenigen Beobachtungen je Arm: `NaN`
- Bei vollständig zensierten Gruppen ohne beobachteten Breach: `NaN`

Hinweis:
- Kaplan-Meier wird hier nicht explizit geplottet; der Log-Rank verwendet dieselbe Survival-Datenlogik mit Zensierung.

## 7.3 Öffentliche API: `compare_conditions(...)`
Führt alle drei Tests aus und liefert ein gemeinsames `ComparisonResult`.

---

## 8. Kopplung zu Persistenz und Runner

### 8.1 Welche Tabellen die Evaluation nutzt
- Primär `stop_events`
- `run_events` für Validierung offener/abgebrochener Runs
- `step_events` werden für ASR/Q2B/T2B aktuell nicht benötigt

### 8.2 Voraussetzung an Runner-Persistenz
`ComparativeRunner` muss pro Run mindestens ein valides `StopEvent` schreiben.
Nur dann sind:
- ASR-Zähler korrekt
- Q2B/T2B-Verteilungen vollständig
- Zensierung korrekt rekonstruiert

### 8.3 Debug-Sidecar
Der Runner schreibt zusätzlich `results/{experiment_id}_debug.jsonl`.
Diese Datei ist für qualitative Analysen nützlich, wird aber vom Evaluation Layer derzeit nicht gelesen.

---

## 9. Testabdeckung (Evaluation Layer)

| Testdatei | Fokus |
|---|---|
| `tests/test_metrics.py` | ASR-Formel, Nullfälle, Condition-/Experiment-Filter, Q2B/T2B inkl. Zensierung |
| `tests/test_stats.py` | Fisher-Verhalten, NaN-Fallback bei zu wenig Daten und vollständig zensierten Gruppen, Survival-p-Werte bei ausreichender Datenlage |
| `tests/test_evaluation_validator.py` | Condition-Matrix, Runzahlen, Seeds, Pairing, Run-Lifecycle, StopEvent-Invarianten |
| `tests/test_evaluation_cli.py` | KM-Median, Holm-Korrektur, `No data`/`n/a`, Validierungsstatus im JSON |

Warum diese Abdeckung wichtig ist:
- schützt vor stillen Fehlern in Zähllogik und Filterung
- sichert korrekte Behandlung rechtszensierter Runs
- stellt robuste Fallback-Semantik bei kleinen Samples sicher

---

## 10. Best-Practice-Alignment (recherchiert)

| Best Practice | Umsetzung im Evaluation Layer | Quelle |
|---|---|---|
| Kleine Stichproben bei Proportionen mit exakten Verfahren testen | ASR via Fisher’s Exact statt asymptotischer Approximation | `scipy.stats.fisher_exact` Doku |
| Rechtszensierung explizit modellieren | `censored_flag` wird bis in Survival-Test durchgereicht | Lifelines Log-Rank/KM Methodik |
| Persistente, reproduzierbare Auswertungspipeline | SQLite EventStore + `experiment_id`-Filter | MARTE-Architektur (`docs/ARCHITECTURE.md`) |
| Defensiver Umgang mit unzureichenden Daten | NaN-Fallback statt pseudo-präziser p-Werte | statistische Good Practice bei insuffizienter Stichprobe |
| Familienweise Fehlerkontrolle pro Metrikfamilie | CLI-Report ergänzt Holm-korrigierte p-Werte für ASR, Q2B und T2B | Standardverfahren für multiple Vergleiche |

---

## 11. Bekannte Grenzen und nächste saubere Schritte

### 11.1 Grenzen (aktueller Stand)
- `StatsService` lädt `stop_events` in `_compare_asr` derzeit zweimal (funktional korrekt, aber nicht optimal).
- Keine Konfidenzintervalle (nur p-Werte/Odds Ratio).
- Keine Effektstärke für Survival jenseits des Log-Rank-p-Werts.
- Holm-Korrektur lebt in der CLI-Report-Pipeline, nicht im `StatsService` selbst.

### 11.2 Saubere nächste Schritte
- internen Doppel-Read in `_compare_asr` zusammenführen
- zusätzliche Ausgabe: Konfidenzintervalle und robuste Effektstärken
- optionale Kaplan-Meier-Exportfunktion (Tabellen/Plots für Thesis-Abbildungen)

---

## 12. Kurzfazit für den Thesis-Implementierungsabschnitt

Der Evaluation Layer ist so implementiert, dass:
- metrische Kerngrößen (ASR, Q2B, T2B) reproduzierbar aus persistierten Run-Daten berechnet werden,
- rechtszensierte Beobachtungen methodisch korrekt in die Survival-Analyse eingehen,
- unvollständige oder inkonsistente Experimente vor der Statistik sichtbar gemacht werden,
- und statistische Vergleiche zwischen `stm_only` und `rag_ltm` robust als strukturierte Ergebnisse (`ComparisonResult`) bereitstehen.

Damit liefert der Layer die technische Grundlage für den inferenzstatistischen Nachweis des Memory-Effekts in MARTE.
