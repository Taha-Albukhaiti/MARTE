import sqlite3
from pathlib import Path

import pytest

from marte.target.guardrail import GuardrailDecision, GuardrailEngine, GuardrailLevel
from marte.target.model_adapter import (
    BackendConfig,
    MockBackend,
    ModelAdapter,
    backend_config_from_env,
    build_backend,
)
from marte.target.sandbox import SandboxBoundary
from marte.target.tool_broker import DispatchResult, ToolBroker
from marte.target.tools import FileTool, SQLTool, ToolCall

"""
Test-Motivation (MARTE-thesis-worktree):
- ToolBroker ist die "technical execution" Schicht zwischen TargetAgent und den konkreten Tools.
  Er muss Guardrail-Entscheidungen korrekt durchsetzen und gleichzeitig robuste, parsebare
  ToolCall-Records liefern (für Tool-Trace, Oracle und Debugging).
- Wichtig: ToolBroker darf nicht crashen (no-raise), sondern Fehler in ToolCall.error abbilden,
  damit Experimente nicht durch Einzel-Fehler abbrechen und Q2B/ASR verzerren.
"""

REPO_ROOT = Path(__file__).parent.parent
SEED_SQL = REPO_ROOT / "sandbox_data" / "sql" / "seed.sql"
SEED_FS = REPO_ROOT / "sandbox_data" / "fs" / "seed"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sandbox(tmp_path: Path) -> SandboxBoundary:
    sb = SandboxBoundary(SEED_SQL, SEED_FS, tmp_path / "runtime")
    sb.reset_sandbox()
    return sb


@pytest.fixture
def moderate_broker(sandbox: SandboxBoundary) -> ToolBroker:
    return ToolBroker(
        guardrail=GuardrailEngine(GuardrailLevel.MODERATE),
        sql_tool=SQLTool(sandbox.db_path),
        file_tool=FileTool(sandbox.fs_path),
    )


@pytest.fixture
def enterprise_broker(sandbox: SandboxBoundary) -> ToolBroker:
    return ToolBroker(
        guardrail=GuardrailEngine(GuardrailLevel.ENTERPRISE),
        sql_tool=SQLTool(sandbox.db_path),
        file_tool=FileTool(sandbox.fs_path),
    )


# ===========================================================================
# DispatchResult structure
# ===========================================================================


class TestDispatchResultStructure:
    """DispatchResult ist die standardisierte Schnittstelle für Tool-Ausführung.

    Warum: Downstream-Komponenten (TargetAgent/Oracle/EventStore) erwarten konsistente Felder
    (policy_decision + tool_call). Ein wackeliger Shape führt zu schwer debugbaren Metrikfehlern.
    """
    def test_allowed_result_has_allow_decision(
        self, moderate_broker: ToolBroker
    ) -> None:
        r = moderate_broker.dispatch_sql("SELECT id FROM users")
        assert r.policy_decision.action == "allow"
        assert r.tool_call.error is None

    def test_denied_result_has_deny_decision(
        self, enterprise_broker: ToolBroker
    ) -> None:
        r = enterprise_broker.dispatch_sql("INSERT INTO users VALUES (99,'X','e',0)")
        assert r.policy_decision.action == "deny"

    def test_denied_result_tool_call_records_reason(
        self, enterprise_broker: ToolBroker
    ) -> None:
        r = enterprise_broker.dispatch_sql("INSERT INTO users VALUES (99,'X','e',0)")
        assert r.tool_call.error is not None
        assert "[DENIED]" in r.tool_call.error

    def test_denied_result_has_empty_output(
        self, enterprise_broker: ToolBroker
    ) -> None:
        r = enterprise_broker.dispatch_sql("DROP TABLE users")
        assert r.tool_call.output == ""

    def test_returns_dispatch_result_type(
        self, moderate_broker: ToolBroker
    ) -> None:
        r = moderate_broker.dispatch_sql("SELECT * FROM users")
        assert isinstance(r, DispatchResult)


# ===========================================================================
# Safety behavior — sanitize and no-raise guarantees
# ===========================================================================


class TestToolBrokerSafety:
    """Safety/Robustness: sanitize, missing tool wiring, Exceptions.

    Warum: Insecure Tool Use ist das Kernthema. ToolBroker muss Guardrail-Aktionen (deny/sanitize)
    *vor* Tool-Execution durchsetzen und darf dabei weder Side-Effects erzeugen noch crashen.
    """
    def test_sanitize_blocks_sql_execution(self) -> None:
        class SanitizingGuardrail:
            def check_tool_call(self, tool_name, tool_input, operation=None):
                return GuardrailDecision(action="sanitize", reason="sanitize required")

        class SpySQLTool:
            def __init__(self) -> None:
                self.called = False

            def execute(self, query: str) -> ToolCall:
                self.called = True
                return ToolCall(
                    tool="sql",
                    operation="SELECT",
                    target="users",
                    raw_input=query,
                    output="[]",
                )

        spy = SpySQLTool()
        broker = ToolBroker(guardrail=SanitizingGuardrail(), sql_tool=spy)
        r = broker.dispatch_sql("SELECT * FROM users")

        assert r.policy_decision.action == "sanitize"
        assert r.tool_call.error is not None
        assert "[SANITIZE]" in r.tool_call.error
        assert spy.called is False

    def test_missing_sql_tool_is_captured_not_raised(self) -> None:
        broker = ToolBroker(guardrail=GuardrailEngine(GuardrailLevel.MODERATE))
        r = broker.dispatch_sql("SELECT * FROM users")
        assert r.policy_decision.action == "deny"
        assert "not configured" in r.policy_decision.reason.lower()
        assert r.tool_call.error is not None
        assert "[DENIED]" in r.tool_call.error

    def test_missing_file_tool_is_captured_not_raised(self) -> None:
        broker = ToolBroker(guardrail=GuardrailEngine(GuardrailLevel.MODERATE))
        r = broker.dispatch_file_read("public/readme.txt")
        assert r.policy_decision.action == "deny"
        assert "not configured" in r.policy_decision.reason.lower()
        assert r.tool_call.error is not None
        assert "[DENIED]" in r.tool_call.error

    def test_missing_file_tool_write_is_captured_not_raised(self) -> None:
        broker = ToolBroker(guardrail=GuardrailEngine(GuardrailLevel.MODERATE))
        r = broker.dispatch_file_write("public/out.txt", "x")
        assert r.policy_decision.action == "deny"
        assert "not configured" in r.policy_decision.reason.lower()
        assert r.tool_call.error is not None
        assert "[DENIED]" in r.tool_call.error

    def test_tool_exception_is_captured_not_raised(self) -> None:
        class FailingSQLTool:
            def execute(self, query: str) -> ToolCall:
                raise RuntimeError("boom")

        broker = ToolBroker(
            guardrail=GuardrailEngine(GuardrailLevel.MODERATE),
            sql_tool=FailingSQLTool(),
        )
        r = broker.dispatch_sql("SELECT * FROM users")
        assert r.policy_decision.action == "allow"
        assert r.tool_call.error is not None
        assert "[ERROR]" in r.tool_call.error

    def test_sanitize_blocks_file_execution(self) -> None:
        class SanitizingGuardrail:
            def check_tool_call(self, tool_name, tool_input, operation=None):
                return GuardrailDecision(action="sanitize", reason="sanitize required")

        class SpyFileTool:
            def __init__(self) -> None:
                self.read_called = False

            def read(self, path: str) -> ToolCall:
                self.read_called = True
                return ToolCall(
                    tool="file",
                    operation="read",
                    target=path,
                    raw_input=path,
                    output="content",
                )

        spy = SpyFileTool()
        broker = ToolBroker(guardrail=SanitizingGuardrail(), file_tool=spy)
        r = broker.dispatch_file_read("public/readme.txt")

        assert r.policy_decision.action == "sanitize"
        assert r.tool_call.error is not None
        assert "[SANITIZE]" in r.tool_call.error
        assert spy.read_called is False


# ===========================================================================
# SQL dispatch — moderate
# ===========================================================================


class TestModerateSQL:
    """Moderate: SQL-Dispatch blockiert offensichtliche geschützte Zugriffe.

    Warum: Das schwächere Niveau soll direkte Angriffe abfangen, ohne die
    umfassendere Enterprise-Prüfung vorwegzunehmen.
    """
    def test_select_returns_rows(self, moderate_broker: ToolBroker) -> None:
        r = moderate_broker.dispatch_sql("SELECT id, name FROM users ORDER BY id")
        assert r.policy_decision.action == "allow"
        assert "Alice" in r.tool_call.output

    def test_select_operation_and_target(self, moderate_broker: ToolBroker) -> None:
        r = moderate_broker.dispatch_sql("SELECT * FROM users")
        assert r.tool_call.operation == "SELECT"
        assert r.tool_call.target == "users"

    def test_insert_allowed_and_persisted(
        self, moderate_broker: ToolBroker, sandbox: SandboxBoundary
    ) -> None:
        r = moderate_broker.dispatch_sql(
            "INSERT INTO users VALUES (99, 'Newbie', 'employee', 30000)"
        )
        assert r.policy_decision.action == "allow"
        conn = sqlite3.connect(sandbox.db_path)
        row = conn.execute("SELECT name FROM users WHERE id=99").fetchone()
        conn.close()
        assert row == ("Newbie",)

    def test_update_allowed(self, moderate_broker: ToolBroker) -> None:
        r = moderate_broker.dispatch_sql("UPDATE users SET salary=0 WHERE id=1")
        assert r.policy_decision.action == "allow"
        assert r.tool_call.error is None

    def test_select_star_from_credentials_blocked(self, moderate_broker: ToolBroker) -> None:
        # Moderate now blocks obvious wildcard reads on credentials
        r = moderate_broker.dispatch_sql("SELECT * FROM credentials")
        assert r.policy_decision.action == "deny"

    def test_alias_wildcard_on_credentials_blocked(self, moderate_broker: ToolBroker) -> None:
        r = moderate_broker.dispatch_sql("SELECT c.* FROM credentials c")
        assert r.policy_decision.action == "deny"

    def test_quoted_credentials_identifier_allowed(self, moderate_broker: ToolBroker) -> None:
        r = moderate_broker.dispatch_sql(
            'SELECT c.user_id FROM "credentials" c ORDER BY c.id'
        )
        assert r.policy_decision.action == "allow"
        assert r.tool_call.error is None

    def test_invalid_sql_captured(self, moderate_broker: ToolBroker) -> None:
        r = moderate_broker.dispatch_sql("NOT VALID SQL $$")
        assert r.policy_decision.action == "allow"   # guardrail passes it through
        assert r.tool_call.error is not None          # SQLite rejects it


# ===========================================================================
# SQL dispatch — enterprise
# ===========================================================================


class TestEnterpriseSQL:
    """Enterprise: SQL-Dispatch blockt gefährliche Operationen und protected-data reads.

    Warum: Enterprise ist die härtere Guardrail-Bedingung. Für korrekte Evaluation muss "deny"
    zuverlässig Side-Effects verhindern (DB-Hash bleibt gleich).
    """
    def test_allows_safe_select(self, enterprise_broker: ToolBroker) -> None:
        r = enterprise_broker.dispatch_sql("SELECT name, role FROM users")
        assert r.policy_decision.action == "allow"
        assert r.tool_call.error is None

    def test_denies_insert(self, enterprise_broker: ToolBroker) -> None:
        r = enterprise_broker.dispatch_sql(
            "INSERT INTO users VALUES (99, 'X', 'employee', 1)"
        )
        assert r.policy_decision.action == "deny"

    def test_denies_update(self, enterprise_broker: ToolBroker) -> None:
        r = enterprise_broker.dispatch_sql("UPDATE users SET salary=0")
        assert r.policy_decision.action == "deny"

    def test_denies_delete(self, enterprise_broker: ToolBroker) -> None:
        r = enterprise_broker.dispatch_sql("DELETE FROM users WHERE id=1")
        assert r.policy_decision.action == "deny"

    def test_denies_drop_table(self, enterprise_broker: ToolBroker) -> None:
        r = enterprise_broker.dispatch_sql("DROP TABLE users")
        assert r.policy_decision.action == "deny"

    def test_denies_select_on_credentials(self, enterprise_broker: ToolBroker) -> None:
        r = enterprise_broker.dispatch_sql("SELECT * FROM credentials")
        assert r.policy_decision.action == "deny"

    def test_denied_insert_does_not_modify_db(
        self, enterprise_broker: ToolBroker, sandbox: SandboxBoundary
    ) -> None:
        snap_before = sandbox.snapshot()
        enterprise_broker.dispatch_sql(
            "INSERT INTO users VALUES (99, 'X', 'employee', 1)"
        )
        snap_after = sandbox.snapshot()
        assert snap_before.db_hash == snap_after.db_hash

    def test_denied_drop_does_not_modify_db(
        self, enterprise_broker: ToolBroker, sandbox: SandboxBoundary
    ) -> None:
        snap_before = sandbox.snapshot()
        enterprise_broker.dispatch_sql("DROP TABLE users")
        snap_after = sandbox.snapshot()
        assert snap_before.db_hash == snap_after.db_hash


# ===========================================================================
# File dispatch — moderate
# ===========================================================================


class TestModerateFile:
    """Moderate: Filesystem-Dispatch erlaubt Reads/Writes frei.

    Warum: Baseline für File-Szenario — erlaubt es, Angriffe tatsächlich bis zur Tool-Wirkung zu
    treiben (und dadurch Unterschiede in Angriffserfolg messbar zu machen).
    """
    def test_read_public_file(self, moderate_broker: ToolBroker) -> None:
        r = moderate_broker.dispatch_file_read("public/readme.txt")
        assert r.policy_decision.action == "allow"
        assert "Welcome" in r.tool_call.output

    def test_read_protected_file_direct_blocked(self, moderate_broker: ToolBroker) -> None:
        # Moderate blocks direct access to protected/secret.txt
        r = moderate_broker.dispatch_file_read("protected/secret.txt")
        assert r.policy_decision.action == "deny"

    def test_read_protected_file_path_variation_allowed(self, moderate_broker: ToolBroker) -> None:
        # Creative path variation passes moderate; oracle fires on normalized path
        r = moderate_broker.dispatch_file_read("./protected/secret.txt")
        assert r.policy_decision.action == "allow"
        assert "CONFIDENTIAL" in r.tool_call.output

    def test_write_to_any_path_allowed(
        self, moderate_broker: ToolBroker, sandbox: SandboxBoundary
    ) -> None:
        r = moderate_broker.dispatch_file_write("protected/out.txt", "hello")
        assert r.policy_decision.action == "allow"
        assert (sandbox.fs_path / "protected" / "out.txt").read_text() == "hello"

    def test_read_operation_and_target(self, moderate_broker: ToolBroker) -> None:
        r = moderate_broker.dispatch_file_read("public/readme.txt")
        assert r.tool_call.operation == "read"
        assert r.tool_call.target == "public/readme.txt"


# ===========================================================================
# File dispatch — enterprise
# ===========================================================================


class TestEnterpriseFile:
    """Enterprise: Filesystem-Dispatch erlaubt Public, blockt Protected und Traversal.

    Warum: Enterprise soll "least privilege" für Dateizugriffe erzwingen; speziell Traversal ist
    eine klassische Escalation-Technik und muss deterministisch geblockt werden.
    """
    def test_allows_read_public(self, enterprise_broker: ToolBroker) -> None:
        r = enterprise_broker.dispatch_file_read("public/readme.txt")
        assert r.policy_decision.action == "allow"
        assert r.tool_call.error is None

    def test_denies_read_protected(self, enterprise_broker: ToolBroker) -> None:
        r = enterprise_broker.dispatch_file_read("protected/secret.txt")
        assert r.policy_decision.action == "deny"

    def test_denied_read_returns_no_content(
        self, enterprise_broker: ToolBroker
    ) -> None:
        r = enterprise_broker.dispatch_file_read("protected/secret.txt")
        assert r.tool_call.output == ""

    def test_allows_write_to_public(
        self, enterprise_broker: ToolBroker, sandbox: SandboxBoundary
    ) -> None:
        r = enterprise_broker.dispatch_file_write("public/new.txt", "safe content")
        assert r.policy_decision.action == "allow"
        assert (sandbox.fs_path / "public" / "new.txt").read_text() == "safe content"

    def test_denies_write_to_protected(self, enterprise_broker: ToolBroker) -> None:
        r = enterprise_broker.dispatch_file_write("protected/out.txt", "evil")
        assert r.policy_decision.action == "deny"

    def test_denied_write_does_not_create_file(
        self, enterprise_broker: ToolBroker, sandbox: SandboxBoundary
    ) -> None:
        snap_before = sandbox.snapshot()
        enterprise_broker.dispatch_file_write("protected/injected.txt", "evil")
        snap_after = sandbox.snapshot()
        assert snap_before.fs_hash == snap_after.fs_hash

    def test_denies_path_traversal(self, enterprise_broker: ToolBroker) -> None:
        r = enterprise_broker.dispatch_file_read("../etc/passwd")
        assert r.policy_decision.action == "deny"


# ===========================================================================
# ModelAdapter
# ===========================================================================


class TestModelAdapter:
    """ModelAdapter/MockBackend stützen reproduzierbare Tests ohne echte API.

    Warum: Das Experiment kontrolliert Varianz (Seeds, deterministische Oracle-Checks). Für Unit-
    Tests brauchen wir einen stabilen LLM-Ersatz, der Calls/kwargs zuverlässig aufzeichnet.
    """
    def test_mock_returns_configured_response(self) -> None:
        adapter = ModelAdapter(MockBackend("hello world"))
        result = adapter.complete([{"role": "user", "content": "hi"}])
        assert result == "hello world"

    def test_mock_records_calls(self) -> None:
        backend = MockBackend("ok")
        adapter = ModelAdapter(backend)
        msgs = [{"role": "user", "content": "test"}]
        adapter.complete(msgs)
        adapter.complete(msgs)
        assert len(backend.calls) == 2

    def test_mock_call_stores_messages(self) -> None:
        backend = MockBackend("ok")
        adapter = ModelAdapter(backend)
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
        adapter.complete(msgs)
        assert backend.calls[0] == msgs

    def test_model_cfg_forwarded_as_kwargs(self) -> None:
        received_kwargs: dict = {}

        class CapturingBackend:
            def complete(self, messages, **kwargs):
                received_kwargs.update(kwargs)
                return "x"

        adapter = ModelAdapter(CapturingBackend(), model_cfg={"temperature": 0, "seed": 42})
        adapter.complete([])
        assert received_kwargs == {"temperature": 0, "seed": 42}

    def test_empty_model_cfg_by_default(self) -> None:
        adapter = ModelAdapter(MockBackend())
        assert adapter.model_cfg == {}

    def test_multiple_different_responses(self) -> None:
        responses = iter(["first", "second", "third"])

        class SequentialBackend:
            def complete(self, messages, **kwargs):
                return next(responses)

        adapter = ModelAdapter(SequentialBackend())
        assert adapter.complete([]) == "first"
        assert adapter.complete([]) == "second"
        assert adapter.complete([]) == "third"


class TestBackendFactory:
    def test_env_use_uni_llm_false_selects_mock(self) -> None:
        cfg = backend_config_from_env({"USE_UNI_LLM": "0"})
        assert cfg.provider == "mock"

    def test_env_defaults_to_openai_compatible_with_mylab_url(self) -> None:
        cfg = backend_config_from_env({})
        assert cfg.provider == "openai_compatible"
        assert cfg.base_url == "https://models.mylab.th-luebeck.dev/v1"

    def test_build_backend_mock_uses_configured_response(self) -> None:
        backend = build_backend(BackendConfig(provider="mock", mock_response="x"))
        assert isinstance(backend, MockBackend)
        assert backend.complete([]) == "x"

    def test_build_backend_openai_requires_key_when_not_compatible(self) -> None:
        cfg = BackendConfig(provider="openai", base_url=None, api_key=None)
        with pytest.raises(ValueError, match="No API key configured"):
            build_backend(cfg)

    def test_env_rejects_unsupported_provider(self) -> None:
        with pytest.raises(ValueError, match="unsupported LLM_PROVIDER"):
            backend_config_from_env({"LLM_PROVIDER": "unsupported_provider"})
