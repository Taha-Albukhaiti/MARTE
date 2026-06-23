# Target Layer — Implementierungsdokumentation

## 1. Zweck und Systemgrenze

Der Target Layer ist das **System Under Test (SUT)** für MARTE.  
Er simuliert ein tool-integriertes LLM-System, das von einem separaten Attacker-Layer (Planner, Mutator, Executor, Judge) angegriffen wird.

**Ziel des Layers:** reproduzierbar ausführen, was ein Target-Agent auf einen Angriffsprompt tut, und dabei alle sicherheitsrelevanten Effekte strukturiert zurückmelden.

**Wichtige Grenze (bewusst so gebaut):**
- Der Target Layer kennt **nicht** den BreachOracle, den EventStore oder die Orchestrator-Logik.
- Der Attacker kommuniziert nur über die technische API (`TargetAgent.execute_step()` via TargetExecutor).

Warum diese Grenze: In der 2x2x2-Evaluation muss das SUT als unabhängiger Faktor stabil bleiben. Eine Kopplung an die Experimentlogik würde Messungen (ASR, Q2B, Time-to-Breach) verfälschen.

---

## 2. Rolle im Multi-Agenten-System

Im Gesamtaufbau gibt es zwei getrennte Agentensichten:

- **Attacker-MAS**: plant, erzeugt Prompts, klassifiziert Ergebnisse.
- **Target-Agent**: verarbeitet den Prompt, ruft ggf. Tools auf, wendet Guardrails an.

Der Target Layer ist damit **nicht** selbst ein kooperatives MAS, sondern eine klar abgegrenzte Gegenstelle im Experiment.

Das ist absichtlich: Für eine saubere Kausalanalyse soll nur der Attacker variieren (z. B. Memory-Modus), während das Target stabil und kontrolliert bleibt.

---

## 3. Designprinzipien (mit Begründung)

### 3.1 Deterministische Zustandsbeobachtung
- `SandboxBoundary.snapshot()` hasht DB- und FS-Zustand deterministisch.
- Breach-Entscheidungen basieren damit auf Systemwirkung statt Textheuristik.

Warum: Das Forschungsziel bewertet Insecure Tool Use als Zustandsverletzung, nicht nur als problematische Antwortformulierung.

### 3.2 Fail-safe statt Crash
- Tools, Broker und Agent kapseln Fehler in strukturierten Ergebnissen (`ToolCall.error`, `[LLM ERROR] ...`).
- Unkontrollierte Exceptions sollen den Run nicht abbrechen.

Warum: Einzelne Backend-Fehler dürfen experimentelle Zeitreihen (Q2B/Survival) nicht zerstören.

### 3.3 Least Privilege und Defense-in-Depth
- Guardrails greifen auf Input-, Tool- und Output-Ebene.
- ToolBroker erzwingt Guardrail-Entscheidungen vor jeder Toolausführung.

Warum: Prompt-Injection kann an unterschiedlichen Stellen wirken; nur eine Kontrollstufe reicht nicht.

### 3.4 Klare Verträge statt implizitem Zustand
- Pydantic-Modelle (`SandboxSnapshot`, `ToolCall`, `GuardrailDecision`) und dataclasses (`StepResult`, `DispatchResult`) bilden stabile Interfaces.
- `Protocol`-basierte Backends erlauben austauschbare LLM-Provider ohne Layer-Bruch.

Warum: Die Implementierung muss in Tests, Logging und späterer Persistenz konsistent interpretierbar sein.

### 3.5 Konfiguration über Environment
- Backend-Auswahl, Timeouts und Retry-Parameter laufen über `BackendConfig` + Env.

Warum: Gleicher Code soll lokal, auf Uni-Endpoint oder mit Mock reproduzierbar laufen.

---

## 4. Komponentenübersicht

| Komponente | Datei | Verantwortung |
|---|---|---|
| `SandboxBoundary`, `SandboxSnapshot` | `sandbox.py` | Reset + Snapshot von Laufzeit-DB/FS |
| `SQLTool` | `tools/sql_tool.py` | SQL ausführen, strukturierte ToolCall-Rückgabe |
| `FileTool` | `tools/file_tool.py` | Sandbox-Dateien lesen/schreiben mit Pfadgrenzen |
| `ToolCall` | `tools/tool_call.py` | Standardisiertes Tool-Ereignis |
| `GuardrailEngine`, `GuardrailDecision`, `GuardrailLevel` | `guardrail.py` | Policy-Prüfung auf Prompt/Tool/Output |
| `ToolBroker`, `DispatchResult` | `tool_broker.py` | Guardrail-Policy durchsetzen und an Tools dispatchen |
| `BackendConfig`, `LLMBackend`, `ModelAdapter`, `OpenAIClientBackend`, `MockBackend`, `RetryingBackend` | `model_adapter.py` | LLM-Abstraktion + Provider-Fabrik |
| `TargetAgent`, `StepResult` | `agent.py` | Fassade, die den gesamten Step orchestriert |
| Re-Exports | `__init__.py` | Stabile Importoberfläche für Außenlayer |

---

## 5. Laufzeitfluss eines Steps

```
Attacker/Executor -> TargetAgent.execute_step(prompt)
  -> Guardrail input check
  -> LLM completion (erste Antwort)
  -> optional Tool-Dispatch (JSON in Antwort)
  -> optional zweiter LLM-Call mit Tool-Feedback
  -> Guardrail output check
  -> StepResult (response_text, tool_trace, policy_decisions, timings, ...)
```

**Warum diese Reihenfolge:**
- Erst Input-Check, damit verbotene Prompts keine Toolwirkung erzeugen.
- Tool-Dispatch vor Output-Filter, weil die Systemwirkung bereits im Toolpfad entsteht.
- Output-Filter separat, damit Datenlecks erfasst werden können, ohne Tool-Policy-Logs zu verfälschen.

---

## 6. Öffentliche Verträge und Invarianten

### 6.1 `SandboxSnapshot` (`sandbox.py`)
- Felder: `db_hash: str`, `fs_hash: str`
- Bedeutet: deterministischer Fingerabdruck des aktuellen Sandbox-Zustands.

### 6.2 `ToolCall` (`tools/tool_call.py`)
- Felder: `tool`, `operation`, `target`, `raw_input`, `output`, `error`
- Bedeutet: vollständiges Audit-Record eines Tool-Aufrufs oder Blockings.

### 6.3 `DispatchResult` (`tool_broker.py`)
- Felder: `tool_call`, `policy_decision`
- Bedeutet: bindet technische Ausführung und Policy-Entscheidung 1:1 zusammen.

### 6.4 `StepResult` (`agent.py`)
- Kernfelder: `response_text`, `tool_trace`, `policy_decisions`, `input_decision`, `output_decision`, `timings`
- Invarianten:
  - `len(policy_decisions) == len(tool_trace)`
  - `timings` enthält immer `llm_ms`, `guardrail_ms`, `tool_ms`
  - Ergebnisobjekt bleibt vollständig, auch bei Fehlern

Warum diese Invarianten: Sie vereinfachen spätere Auswertung und verhindern implizite Sonderfälle im Attack-Layer.

---

## 7. Komponenten im Detail

## 7.1 SandboxBoundary (`sandbox.py`)

### Was es macht
- `reset_sandbox()`: setzt `runtime/sandbox.db` und `runtime/fs` auf Seed-Zustand zurück.
- `snapshot()`: liefert Hashes für DB und FS.

### Wie DB-Hash funktioniert
- liest `sqlite_master`-Objekte (Schemaebene),
- hasht Spaltendefinitionen (`PRAGMA table_info`),
- hasht alle Zeilen pro Tabelle in deterministischer Reihenfolge.

Spezialfall: `WITHOUT ROWID`-Tabellen werden über Primary-Key-Reihenfolge stabil geordnet, nicht mit `ORDER BY rowid`.

### Warum so
- SQLite-Dateibytes sind kein stabiler Inhaltsindikator (z. B. Journaling/Pages), deshalb inhaltsbasiertes Hashing.
- Schemaänderungen müssen ebenso erkannt werden wie Datenänderungen.

---

## 7.2 SQLTool (`tools/sql_tool.py`)

### Was es macht
- führt genau ein SQL-Statement aus (oder mehrere via `executescript`),
- klassifiziert `operation` per Regex,
- extrahiert `target` (Tabelle/Objekt) heuristisch.

### Warum so
- Der Oracle braucht strukturierte Felder (`operation`, `target`) auch bei SQL-Fehlern.
- Fehler werden in `ToolCall.error` gekapselt (no-raise), damit Runs stabil bleiben.

### Bewusste Grenze
- Parser ist heuristisch (Regex), nicht SQL-AST-basiert.
- Für PoC/Benchmark ausreichend, für produktive SQL-Sicherheitsanalyse nicht vollständig.

---

## 7.3 FileTool (`tools/file_tool.py`)

### Was es macht
- normalisiert virtuelle Pfade (`PurePosixPath`),
- mappt absolute virtuelle Pfade (`/protected/secret.txt`) auf Sandbox-relative Pfade,
- blockiert Pfad-Escapes via `resolve()` + `is_relative_to()`.

### Warum so
- verhindert Host-Filesystem-Zugriffe trotz manipuliertem Pfadinput.
- unterstützt gleichzeitig realistisches Agentenverhalten (absolute Pfade als Sandbox-Referenz).

---

## 7.4 GuardrailEngine (`guardrail.py`)

### Was es macht
- `moderate` und `enterprise` Level; `moderate` blockt offensichtliche Klartextangriffe und direkte Tool-JSON-Erzwingung,
- `check_input()`, `check_tool_call()`, `check_output()`,
- enterprise blockt zusätzlich obfuskierte Angriffsformen, geschützte SQL-Tabellen und sensible Dateipfade.

### Warum so
- Mehrstufige Guardrail-Prüfung deckt unterschiedliche Angriffsoberflächen ab.
- Zwei Levels sind nötig, um in der Forschungsfrage Stabilität über Guardrail-Niveaus zu testen.

### SQL-spezifisch
- enterprise erlaubt nur SELECT-ähnliche Tool-Operationen,
- blockt protected-table-Zugriffe (`credentials`) auch in JOIN/Subquery-Kontexten.

### File-spezifisch
- blockt Traversal,
- blockt Reads unter `protected/`,
- erlaubt Writes nur unter `public/`.

---

## 7.5 ToolBroker (`tool_broker.py`)

### Was es macht
- zentrale Dispatch-Schicht für SQL/File,
- wendet Guardrail-Entscheidung vor Tool-Execution an,
- erzeugt bei `deny/sanitize` einen strukturierten Block-`ToolCall`.

### Warum so
- Eine zentrale Policy-Enforcement-Stelle reduziert Inkonsistenzen zwischen Tools.
- Blocking ohne Toolausführung stellt sicher, dass deny/sanitize keine Side-Effects erzeugen.

---

## 7.6 ModelAdapter + Backend-Fabrik (`model_adapter.py`)

### Was es macht
- `LLMBackend` als Protocol-Vertrag,
- konkrete Backends: `OpenAIClientBackend`, `MockBackend`,
- Wrapper `RateLimitedBackend` (globaler Mindestabstand zwischen API-Aufrufen),
  `RetryingBackend` (exponential backoff) und `AbortOnBackendError`
  (Übergabe ausgeschöpfter Backendfehler an die Phasensteuerung),
- strukturierte Antwortfehler für Trunkierung und inhaltslose Provider-Antworten,
- `backend_config_from_env()` + `build_backend()` erzeugen Backend aus Env.

### Warum so
- Provider-Wechsel (Uni API, OpenAI-kompatibel, Mock) ohne Codeänderung.
- Rate-Limit- und Retry-Logik bleiben Infrastruktur-Concerns statt in Business-Logik verteilt.
- Attack- und Target-Seite können getrennte Modelle/Configs nutzen.

---

## 7.7 TargetAgent (`agent.py`)

### Was es macht
- `init_config(...)` setzt run-spezifische Guardrails, Tool-Allowlist, Model-Config.
- `execute_step(prompt)` führt den kompletten Step aus und liefert `StepResult`.

### Zentrale Entscheidungen
- **JSON-Tool-Call Parsing mit `json.JSONDecoder.raw_decode()`** statt flacher Regex:
  robust bei eingebettetem Text und verschachtelten JSON-Inhalten.
- **Output-Decision separat von `policy_decisions`**:
  verhindert, dass Output-Block den Tool-Policy-Audittrail überschreibt.
- **Optionaler Tool-Feedback-Loop**:
  erlaubt zweiten LLM-Call mit Toolergebnis, ohne den Standardpfad zu erzwingen.

### Warum so
- Fassade hält Außen-API minimal und stabil.
- Interne Helfer (`_timed_check_input`, `_record_tool_dispatch`, ...) halten den Step-Ablauf lesbar und testbar.

---

## 8. Fehler- und Sicherheitsmodell

### 8.1 Fehlerbehandlung
- Ziel: technische Providerfehler innerhalb derselben logischen Phase behandeln,
  ohne den wissenschaftlichen Schrittzähler oder das Gedächtnis zu verändern.
- Fehlerkanäle:
  - LLM: `response_text = [LLM ERROR] ...`
  - LLM-Trunkierung: `LLMCompletionTruncatedError` mit Rolle, Modell, Tokenlimit und Usage; Wiederholung derselben Phase
  - leere abgeschlossene LLM-Antwort: `LLMEmptyResponseError`
  - Tool: `ToolCall.error`
  - Policy-Block: `[DENIED]` / `[SANITIZE]` Marker

Trunkierte oder inhaltslose Antworten werden bis zur zuständigen Phasensteuerung
propagiert. Planner, Mutator, initialer Zielaufruf und Judge werden jeweils direkt
wiederholt. Nach einer bereits ausgeführten Werkzeugoperation verbleibt die
Werkzeugspur unverändert; nur der zweite Zielmodellaufruf wird erneut angefragt.
Der technische Fehlversuch verbraucht kein Q2B-Budget und wird nicht im LTM
gespeichert. HTTP-Fehler, die eindeutig auf eine permanente Fehlkonfiguration
hinweisen, beispielsweise ein unbekannter Modellname, werden nicht wiederholt.

### 8.2 Sicherheitsziele im Layer
- keine Toolausführung bei deny/sanitize,
- keine Host-Pfad-Escapes über FileTool,
- nachvollziehbarer Policy-Trail pro Tool-Call,
- Snapshot-basierte Nachweisbarkeit von Systemwirkung.

---

## 9. Teststrategie (Ist-Stand)

Die Target-Komponenten sind über folgende Dateien abgedeckt:

| Testdatei | Fokus |
|---|---|
| `tests/test_sandbox.py` | Reset-Determinismus, Hash-Stabilität, Schema-/Datenänderungen, WITHOUT ROWID |
| `tests/test_tools.py` | SQL/File Tool-Verhalten, no-raise, Traversal-Schutz, Snapshot-Integration |
| `tests/test_guardrail.py` | moderate vs enterprise, Obfuskationen, SQL/File-Policy, Output-Filter |
| `tests/test_tool_broker.py` | Policy-Enforcement, deny/sanitize ohne Side-Effects, robuste Fehlerkapselung |
| `tests/test_target_agent.py` | End-to-End Step-Flow, Invarianten, Reconfig, LLM-Fehler, Regressionen |
| `tests/test_model_adapter_retry.py` | Retry/Backoff-Semantik, Trunkierungsdiagnose, Env-Parsing, Backend-Wrapping |

Warum diese Aufteilung: Jeder Layervertrag wird dort getestet, wo er entsteht (lokale Verantwortung), plus gezielte End-to-End-Regressionen an der Fassade.

---

## 10. Best-Practice-Alignment (recherchiert)

| Best Practice | Umsetzung im Target Layer | Quelle |
|---|---|---|
| Structural Subtyping für austauschbare Implementierungen | `LLMBackend` als `Protocol` + `@runtime_checkable` | [Python typing docs](https://docs.python.org/3.11/library/typing.html), [PEP 544](https://peps.python.org/pep-0544/) |
| Klare Datenverträge statt unstrukturierter Dicts | Pydantic-Modelle (`ToolCall`, `GuardrailDecision`, `SandboxSnapshot`) + dataclasses (`StepResult`) | [Pydantic BaseModel](https://docs.pydantic.dev/latest/api/base_model/), [dataclasses](https://docs.python.org/3.11/library/dataclasses.html) |
| Sichere Pfadkanonisierung gegen Traversal | `resolve()` + `is_relative_to()` in `FileTool` | [pathlib docs](https://docs.python.org/3.11/library/pathlib.html) |
| Robustes JSON-Parsing in eingebettetem Text | `json.JSONDecoder.raw_decode()` statt Regex | [json module docs](https://docs.python.org/3.14/library/json.html) |
| Inhaltsbasierte DB-Analyse statt Dateibytes | Snapshot über Schema + Tabelleninhalte; Behandlung von `WITHOUT ROWID` | [SQLite WITHOUT ROWID](https://www.sqlite.org/withoutrowid.html), [SQLite schema table](https://www.sqlite.org/schematab.html) |
| Mehrstufige Prompt-Injection-Abwehr | Input-, Tool- und Output-Checks in GuardrailEngine | [OWASP LLM01](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) |
| Konfiguration über Environment | `backend_config_from_env()` und role-spezifische Env-Overrides | [Twelve-Factor Config](https://12factor.net/config) |
| Resilienz bei transienten API-Fehlern und Rate-Limits | `RateLimitedBackend` mit Mindestabstand plus `RetryingBackend` mit exponential backoff; nicht retrybare Antwortfehler werden getrennt behandelt | [Google Cloud Retry Guidelines](https://cloud.google.com/storage/docs/retry-strategy) |

Hinweis: Die aktuelle Retry-Implementierung nutzt exponentiellen Backoff ohne Jitter. Für stark parallelisierte Lastszenarien ist Jitter als nächster Schritt sinnvoll.

---

## 11. Bekannte Grenzen und saubere Erweiterungspunkte

### 11.1 Grenzen
- SQL-Operationserkennung ist regex-basiert (kein vollständiger SQL-Parser).
- Guardrails sind regelbasiert; semantische Umgehungen sind prinzipiell möglich.
- Output-Filter arbeitet pattern-basiert, nicht datensensitiv pro Feldschema.
- Retry-Strategie aktuell ohne Jitter. Nach ausgeschöpften Request-Retries
  werden API-/Netzwerkfehler als `LLMBackendUnavailableError` propagiert. Der
  Runner wiederholt daraufhin den vollständigen Lauf mit demselben Seed und
  pausiert erst nach ausgeschöpften Laufversuchen.

### 11.2 Erweiterungspunkte
- neues Tool:
  - Tool-Klasse in `src/marte/target/tools/`
  - Dispatch-Methode in `ToolBroker`
  - Routing in `TargetAgent._dispatch`
  - Guardrail-Regeln + Tests
- neuer LLM-Provider:
  - Backend implementiert `LLMBackend.complete()`
  - Integration in `build_backend()`
  - Env-Mapping ergänzen

---

## 12. Kurzfazit für den Implementierungsabschnitt der Thesis

Der Target Layer ist so umgesetzt, dass er:
- experimentell sauber entkoppelt ist,
- sicherheitsrelevante Wirkung auf DB/FS reproduzierbar messbar macht,
- robuste und auswertbare Step-Artefakte erzeugt,
- und trotzdem als PoC erweiterbar bleibt (neue Tools, neue Provider, härtere Guardrails).

Damit bildet er eine belastbare technische Basis für die spätere Evaluation von Attack-Strategien, Guardrail-Niveaus und Memory-Konfigurationen.
