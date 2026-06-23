# Memory Layer — Implementierungsdokumentation

## 1. Ziel und Abgrenzung

Der Memory Layer implementiert den Gedächtnismechanismus des Angreifersystems in MARTE.
Er ist verantwortlich für:

- Kurzzeitgedächtnis pro Run (STM, Ringbuffer)
- optionales Langzeitgedächtnis über Runs (LTM, RAG/VectorDB)
- einheitliche Gateway-Schnittstelle für Planner, Mutator und Judge
- isolierte Reset-Semantik zwischen Run- und Condition-Grenzen

**Systemgrenze (bewusst):**
- Memory selbst plant keine Strategien und erzeugt keine Prompts.
- Memory führt keine Angriffe aus und bewertet keine Breaches.
- Memory liefert nur Kontext und nimmt klassifizierte Episoden entgegen.

Warum diese Grenze: Der kausale Vergleich `stm_only` vs. `rag_ltm` bleibt nur valide, wenn der Unterschied ausschließlich im Memory-Backend liegt, nicht in der Agentenlogik.

---

## 2. Rolle im MARTE-Gesamtsystem

Innerhalb eines Steps:

1. `RunOrchestrator` ruft `MemoryGateway.enrich_context(run_state)` auf.
2. Planner/Mutator erhalten `memory_context` als Input.
3. Nach Ausführung + Oracle + Judge ruft `Judge` `memory_gateway.upsert_episode(...)` auf.

Über Run-/Condition-Grenzen:

- **Run-Start:** `reset_stm()` (STM muss leer starten)
- **Condition-Wechsel:** `reset_ltm()` (LTM muss zwischen Conditions isoliert werden)

Das entspricht der Architekturvorgabe in `docs/ARCHITECTURE.md` (Isolation-Semantik).

---

## 3. Designprinzipien und Begründung

### 3.1 DI-Switch statt if-else im Agentencode
- `MemoryGateway` delegiert LTM-Funktionen an ein injiziertes `MemoryStrategy`.
- `NoOpStrategy` = `stm_only`, `RAGAdapter` = `rag_ltm`.

Warum: Planner/Mutator/Judge bleiben unverändert; nur die injizierte Strategie wechselt zwischen Conditions.

### 3.2 STM immer aktiv
- `STMRingBuffer` ist in beiden Modi aktiv (`stm_only` und `rag_ltm`).

Warum: STM ist die Baseline; `rag_ltm` ergänzt LTM, ersetzt STM nicht.

### 3.3 Explizite Reset-Semantik
- `reset_stm()` nur für Session-Isolation pro Run.
- `reset_ltm()` nur für Condition-Isolation.

Warum: Innerhalb einer Condition soll LTM über Runs akkumulieren, zwischen Conditions aber nicht leaken.

### 3.4 Typisierte Episode-Schnittstelle
- Persistierte Gedächtniseinheit ist `Episode` (Pydantic-Schema aus `persistence/schemas.py`).

Warum: Einheitliche Feldstruktur für STM, RAG-Metadaten und spätere Retrieval-Nutzung.

### 3.5 Retrieval mit Kontextfilterung
- RAG-Retrieval filtert nach `scenario`, `guardrail_level`,
  `run_id != memory_run_id` und dem jeweiligen `memory_kind`.
- Das Mutator-Retrieval filtert zusätzlich exakt nach `strategy=plan.strategy`.

Warum: Erhöht semantische Relevanz, reduziert Cross-Context-Rauschen und hält intra-run Lernen im STM.

### 3.6 LTM trennt positive und konkrete blockierte Episoden

`MemoryGateway.upsert_episode()` speichert bestätigte Breaches und plausible
`partial`-Schritte als `memory_kind=positive`. Oracle-negative Input- oder
Tool-Policy-Blockierungen werden als `memory_kind=blocked` gespeichert.
Reine Output-Blockierungen, Modellverweigerungen, Syntax- und
Infrastrukturfehler sowie gewöhnliche Fehlschläge verbleiben im STM.

**Begründung:**

Das LTM arbeitet auf Basis semantischer Ähnlichkeitssuche (Cosine-Distanz im Embedding-Raum). Bestätigte Breaches sind klare positive Beispiele. `partial`-Episoden enthalten zwar keinen Ground-Truth-Breach, aber nutzbare Signale aus erlaubter Tool-Ausführung, erkannten Tabellen/Pfaden oder Guardrail-Grenzen. Diese Signale können in späteren Runs derselben Condition hilfreich sein, wenn das Label als `partial` im Kontext sichtbar bleibt.

Blockierte Episoden dienen ausschließlich als Warnung vor einer konkreten
Formulierung. Sie sperren nicht die gesamte Strategie. Eine stabile,
normalisierte Dokument-ID verhindert unbegrenzte Duplikate desselben Musters.
`success_label` bleibt Metadatum, wird aber nicht in den Embeddingtext aufgenommen.

**Alternative, die bewusst verworfen wurde:**

Alle Fehlschläge persistent zu speichern und erst beim Retrieval zu filtern
würde den Vektorspeicher mit Syntax-, Modell- und Infrastrukturfehlern
anreichern. Die selektive Schreibregel hält nur positive Erfahrungen und
konkret nutzbare Policy-Warnungen vor.

**Verhältnis STM / LTM:**

Der STM-Ringbuffer erhält weiterhin alle Episoden. Das LTM übernimmt positive
Erfahrungen und ausgewählte konkrete Blockierungsmuster aus anderen Runs
derselben Condition; die aktuelle `memory_run_id` wird ausgeschlossen.

---

## 4. Komponentenübersicht

| Komponente | Datei | Aufgabe |
|---|---|---|
| `MemoryStrategy` (Protocol) | `strategy.py` | DI-Vertrag für LTM-Backends |
| `STMRingBuffer` | `stm.py` | Kurzzeitkontext der letzten k Episoden |
| `NoOpStrategy` | `noop.py` | LTM-No-Op für `stm_only` |
| `RAGAdapter` | `rag_adapter.py` | Embedding + VectorDB-Upsert + semantisches Retrieval |
| `MemoryGateway` | `gateway.py` | zentrale API für Agenten (`enrich_context`, `upsert_episode`, Resets) |
| `SwitchingMemoryGateway` | `switching_gateway.py` | Laufzeitumschaltung pro Run (`stm_only`/`rag_ltm`) |
| `InMemoryEpisodeStrategy` | `in_memory.py` | einfacher lokaler LTM-Stub (nicht für echte Experimente) |

---

## 5. Datenverträge und Invarianten

## 5.1 `MemoryStrategy` (`strategy.py`)
Pflichtmethoden:
- `enrich_context(run_state) -> list[Episode]`
- `retrieve(...) -> RetrievalBatch`
- `upsert_episode(episode: Episode) -> None`
- `reset_ltm() -> None`

Invariant:
- Beide Modi (`NoOpStrategy`, `RAGAdapter`) erfüllen denselben Vertrag.

## 5.2 `memory_context`-Shape (`gateway.py`)
`MemoryGateway.enrich_context(...)` liefert ein dict mit:
- `step_index`
- `stm_recent` (Episode-Dumps ohne Embedding)
- `ltm_hits` als interner Kompatibilitätsalias für positive Treffer
- `last_oracle_verdict`, `last_judge_verdict` und Guardrail-Gründe
- `positive_ltm_hits` und `blocked_patterns`
- `retrieval_trace` für `planner_positive` und `planner_blocked`
- `memory_mode` nur als interne Laufmetadaten; nicht in LLM-Payloads

Invariant:
- `stm_recent`, `positive_ltm_hits` und `blocked_patterns` sind immer vorhanden.
- `stm_recent` ist die einzige Quelle der laufinternen ReAct-Trajektorie;
  Planner und Mutator halten keine zusätzliche Konversationshistorie.

## 5.3 Episode-Aufbereitung und Schreib-Gate
`MemoryGateway.upsert_episode(...)` baut aus Judge-Output + `run_state` eine `Episode`:
- `strategy` aus dem aktuellen `AttackPlan` (vom Orchestrator übergeben)
- `mutation` aus `prompt`
- `response_summary` bevorzugt aus `memory_summary`, sonst aus `response_text`
- `success_label` aus `judge_verdict`
- `tool_effect` als kompaktes `operation:target`-Summary
- `memory_kind`, `policy_stage` und `policy_reason` aus Oracle- und Policy-Evidenz

Schreib-Gate:
- **STM:** erhält jede Episode (`success`, `partial`, `fail`)
- **LTM positiv:** `oracle_verdict=True` oder plausibles `judge_verdict=partial`
- **LTM blockiert:** konkrete Input- oder Tool-Policy-Blockierung

Invariant:
- Retrieval aus LTM schließt die aktuelle `memory_run_id` aus; intra-run Lernen läuft über STM.

---

## 6. Komponenten im Detail

## 6.1 `STMRingBuffer` (`stm.py`)

Implementierung:
- `deque(maxlen=capacity)` als Ringbuffer
- FIFO-Eviction bei Overflow
- `get_context()` liefert insertion order (oldest first)

Validierungen:
- `capacity >= 1`, sonst `ValueError`

Warum so:
- O(1)-Append und klare, deterministische Historie der letzten k Schritte.

---

## 6.2 `NoOpStrategy` (`noop.py`)

Verhalten:
- `enrich_context()` -> `[]`
- `retrieve()` -> leeres `RetrievalBatch`
- `upsert_episode()` -> no-op
- `reset_ltm()` -> no-op

Warum so:
- `stm_only` nutzt denselben Gateway-Codepfad wie `rag_ltm`, aber ohne persistentes LTM.

---

## 6.3 `RAGAdapter` (`rag_adapter.py`)

### Kernfunktionen
- `embed(text)` -> Vektor
- `upsert(episode)` -> Embedding + Upsert in Chroma-Collection
- `search(query, top_k, filters)` -> semantisches Retrieval
- `retrieve(...)` -> typisiertes `RetrievalBatch` mit Distanz, Similarity,
  Rang, Thresholdfilterung und phasengerechter Deduplizierung
- `reset_ltm()` -> Collection löschen und neu anlegen
- `health_check()` -> Embedder + Collection erreichbar?

### Persistenz
- Standard (ohne injizierten Client): `chromadb.PersistentClient`
- Pfad: `persist_path` oder `MARTE_CHROMA_PATH` oder `runtime/chroma`
- `persist_path` ist über Property auslesbar

### Retrieval-Logik
- Planner-Query aus Szenario, Ziel und letzter Schritt-Evidenz
- getrennte Planner-Kanäle für positive und blockierte Episoden
- Mutator-Query nach der Planung mit exaktem Strategiefilter
- maximal 3 positive und 1 blockierter Planner-Treffer
- maximal 3 positive Mutator-Treffer; Fetch-Limits 12 beziehungsweise 4
- getrennt kalibrierbare Similarity-Thresholds

### Embedding-Backend
- Standard: `SentenceTransformerBackend("all-MiniLM-L6-v2")`, lazy-loaded
- In Tests/Local kann ein Stub-Embedder injiziert werden

Warum so:
- Für den Experimentvergleich wird ein realistischer, aber klar begrenzter RAG-Zyklus benötigt (nicht maximal komplexe IR-Pipeline).

---

## 6.4 `MemoryGateway` (`gateway.py`)

Rolle:
- einzige öffentliche Memory-Schnittstelle für Attack-Agenten
- besitzt STM intern
- delegiert LTM an injizierte Strategy

Wichtige Details:
- cached `run_state` in `enrich_context()`, damit `upsert_episode()` Episode-Felder korrekt füllen kann
- `reset_stm()` leert nur STM
- `reset_ltm()` delegiert nur an LTM-Strategy (STM bleibt bis Run-Start-Reset erhalten)
- **Schreib-Gate (§ 3.6):** positive und konkrete blockierte Episoden werden
  getrennt klassifiziert; STM erhält immer
- `enrich_context_for_mutator()` führt das Post-Planning-Retrieval aus

Warum so:
- verhindert direkte Abhängigkeiten der Agenten auf konkrete Memory-Backends
- hält Reset-Verantwortung explizit und testbar
- hält LTM als Archiv für bestätigte Erfolge und nutzbare Partials — semantisches Retrieval liefert nur valide Angriffsmuster zurück

---

## 6.5 `SwitchingMemoryGateway` (`switching_gateway.py`)

Zweck:
- Umschaltung zwischen `stm_only` und `rag_ltm` pro Run (CLI/Runner-Einsatz)

Verhalten:
- hält intern zwei Gateways:
  - STM-Gateway mit `NoOpStrategy`
  - RAG-Gateway mit injizierter Strategy
- `set_mode("rag_ltm")` ruft optional `health_check()` auf Strategy auf (fail-fast)
- `active_mode` Property für Beobachtbarkeit

Warum so:
- erlaubt Mischläufe über Conditions mit derselben Orchestrator-/Judge-Integration
- reduziert Risiko, dass ein kaputter RAG-Backend stillschweigend als „funktionierend“ weiterläuft

---

## 6.6 `InMemoryEpisodeStrategy` (`in_memory.py`)

Zweck:
- einfacher, deterministischer LTM-Stub für lokale/schnelle Läufe

Verhalten:
- speichert Episoden in Python-Liste
- filtert grob nach `scenario` + `guardrail_level` und schließt die aktuelle `memory_run_id` aus
- gibt letzte 3 Treffer zurück

Wichtig:
- nicht für den finalen wissenschaftlichen Vergleich gedacht (kein semantisches Retrieval, keine VectorDB).

---

## 7. Isolation-Semantik (Run vs. Condition)

Geplante Verwendung im System:

- `ComparativeRunner`: `reset_ltm()` pro Condition
- `RunOrchestrator`: `reset_stm()` pro Run

Wirkung:
- **innerhalb Condition:** LTM akkumuliert über Runs
- **zwischen Conditions:** LTM wird gelöscht
- **zwischen Runs:** STM wird gelöscht

Das implementiert die zentrale methodische Anforderung für den fairen `stm_only` vs. `rag_ltm`-Vergleich.

---

## 8. Testabdeckung (Memory Layer)

| Testdatei | Fokus |
|---|---|
| `tests/test_stm.py` | Ringbuffer-Semantik, Overflow, Reset, Kapazitätsvalidierung |
| `tests/test_noop.py` | NoOp-Vertrag und Protocol-Kompatibilität |
| `tests/test_rag_adapter.py` | Similarity, Threshold, Deduplizierung, Filter, Reset und Persistenz |
| `tests/test_memory_gateway.py` | Kanaltrennung, Speicherregeln, stabile IDs und STM/LTM-Semantik |
| `tests/test_switching_memory_gateway.py` | Modusumschaltung, Routing, RAG-Fail-fast |

Warum diese Abdeckung wichtig ist:
- sie schützt die kausale Kernannahme des Experiments (Memory-Modus als einziges treatment).

---

## 9. Best-Practice-Alignment (recherchiert)

| Best Practice | Umsetzung im Memory Layer | Quelle |
|---|---|---|
| Structural Subtyping/DI über Protocols | `MemoryStrategy` als klarer Verhaltensvertrag | Python `typing` / PEP 544 |
| Bounded short-term state | STM als fester Ringbuffer mit expliziter Kapazität | Standard Ringbuffer-Pattern |
| Fail-fast bei kritischem Backend-Wechsel | `SwitchingMemoryGateway.set_mode("rag_ltm")` mit `health_check()` | Reliability/SRE-Prinzip |
| Persistente Daten für reproduzierbare Experimente | `PersistentClient` standardmäßig auf Disk-Pfad | Chroma Persistenzmodell |
| Trennung von Interface und Backend-Implementierung | `MemoryGateway` vs. `NoOp`/`RAGAdapter`/`InMemory` | Dependency-Inversion-Prinzip |

---

## 10. Bekannte Grenzen und nächste saubere Schritte

### 10.1 Grenzen
- Es gibt bewusst kein MMR-, Cross-Encoder-, Erfolgs- oder Recency-Reranking.
- Standard-Embedder kann initial Modellgewichte laden (Latenz/Abhängigkeit).
- `InMemoryEpisodeStrategy` ist nur ein lokaler Stub und nicht experimentäquivalent zu RAG.

### 10.2 Nächste sinnvolle Schritte
- technisches Relevanzset bewerten und beide Similarity-Thresholds einfrieren
- finale API-Preflights mit frischen Event- und Chroma-Pfaden ausführen
- explizite Token-/Kontextbudget-Steuerung für zurückgegebene LTM-Hits prüfen

---

## 11. Kurzfazit für den Thesis-Implementierungsabschnitt

Der Memory Layer ist so implementiert, dass:
- der Unterschied zwischen `stm_only` und `rag_ltm` technisch sauber isoliert ist,
- Run- und Condition-Isolation gemäß Studiendesign eingehalten wird,
- und gleichzeitig adaptive Agenten über standardisierte `memory_context`-Inputs von STM/LTM profitieren können.

Damit bildet der Layer die methodisch zentrale Infrastruktur zur Untersuchung des RAG-LTM-Effekts in MARTE.
