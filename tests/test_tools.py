import json
import sqlite3
from pathlib import Path

import pytest

from marte.target.sandbox import SandboxBoundary
from marte.target.tools import FileTool, SQLTool, ToolCall

"""
Test-Motivation (MARTE-thesis-worktree):
- SQLTool und FileTool sind die "ground truth" Side-Effect-Mechanik in der Sandbox. MARTE-thesis-worktree misst
  Angriffserfolg über Tool-zentrierte Wirkung (DB/FS Zustandsänderung), nicht nur Text.
- Daher müssen Tools (a) deterministisch arbeiten, (b) nie Exceptions nach außen werfen (no-raise),
  und (c) verständliche ToolCall-Records liefern (tool, operation, target, output/error).
- Zusätzlich: Snapshot-Integration ist kritisch, weil der BreachOracle auf db_hash/fs_hash basiert.
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
def sql(sandbox: SandboxBoundary) -> SQLTool:
    return SQLTool(sandbox.db_path)


@pytest.fixture
def fs(sandbox: SandboxBoundary) -> FileTool:
    return FileTool(sandbox.fs_path)


# ===========================================================================
# SQLTool
# ===========================================================================


class TestSQLToolSelect:
    """SELECTs liefern JSON-Output und korrekte Metadaten (operation/target)."""
    def test_operation_and_target(self, sql: SQLTool) -> None:
        r = sql.execute("SELECT * FROM users")
        assert r.tool == "sql"
        assert r.operation == "SELECT"
        assert r.target == "users"
        assert r.error is None

    def test_returns_all_seed_rows(self, sql: SQLTool) -> None:
        r = sql.execute("SELECT id, name FROM users ORDER BY id")
        rows = json.loads(r.output)
        assert len(rows) == 5
        assert rows[0] == [1, "Alice"]
        assert rows[4] == [5, "Eve"]

    def test_output_is_valid_json(self, sql: SQLTool) -> None:
        r = sql.execute("SELECT * FROM credentials")
        parsed = json.loads(r.output)
        assert isinstance(parsed, list)

    def test_select_empty_result(self, sql: SQLTool) -> None:
        r = sql.execute("SELECT * FROM users WHERE id = 9999")
        assert json.loads(r.output) == []
        assert r.error is None

    def test_target_extracted_with_double_quotes(self, sql: SQLTool) -> None:
        r = sql.execute('SELECT name FROM "users" WHERE id = 1')
        assert r.target == "users"

    def test_select_credentials_target(self, sql: SQLTool) -> None:
        r = sql.execute("SELECT user_id FROM credentials ORDER BY id")
        assert r.target == "credentials"
        assert r.error is None


class TestSQLToolPragma:
    """Read-only schema queries return rows and structured metadata."""

    def test_table_info_returns_schema_rows(self, sql: SQLTool) -> None:
        r = sql.execute("PRAGMA table_info('credentials')")

        assert r.operation == "PRAGMA"
        assert r.target == "credentials"
        assert r.error is None
        rows = json.loads(r.output)
        assert [row[1] for row in rows] == ["id", "user_id", "password_hash"]


class TestSQLToolInsert:
    """Writes müssen persistieren (Side-Effects sind messrelevant für Breach-Definition)."""
    def test_operation_and_target(self, sql: SQLTool) -> None:
        r = sql.execute("INSERT INTO users VALUES (99, 'X', 'employee', 1)")
        assert r.operation == "INSERT"
        assert r.target == "users"
        assert r.error is None

    def test_row_is_persisted(self, sql: SQLTool, sandbox: SandboxBoundary) -> None:
        sql.execute("INSERT INTO users VALUES (99, 'X', 'employee', 1)")
        conn = sqlite3.connect(sandbox.db_path)
        row = conn.execute("SELECT name FROM users WHERE id=99").fetchone()
        conn.close()
        assert row == ("X",)

    def test_rowcount_in_output(self, sql: SQLTool) -> None:
        r = sql.execute("INSERT INTO users VALUES (99, 'X', 'employee', 1)")
        assert "1" in r.output


class TestSQLToolUpdate:
    """UPDATE ist eine Side-Effect-Operation; wir prüfen Persistenz und Metadaten."""
    def test_operation_and_target(self, sql: SQLTool) -> None:
        r = sql.execute("UPDATE users SET salary = 0 WHERE id = 1")
        assert r.operation == "UPDATE"
        assert r.target == "users"
        assert r.error is None

    def test_change_is_persisted(self, sql: SQLTool, sandbox: SandboxBoundary) -> None:
        sql.execute("UPDATE users SET salary = 0 WHERE id = 1")
        conn = sqlite3.connect(sandbox.db_path)
        salary = conn.execute("SELECT salary FROM users WHERE id=1").fetchone()[0]
        conn.close()
        assert salary == 0


class TestSQLToolDelete:
    """DELETE ist eine Side-Effect-Operation; wir prüfen Persistenz und Metadaten."""
    def test_operation_and_target(self, sql: SQLTool) -> None:
        r = sql.execute("DELETE FROM users WHERE id = 1")
        assert r.operation == "DELETE"
        assert r.target == "users"
        assert r.error is None

    def test_row_is_removed(self, sql: SQLTool, sandbox: SandboxBoundary) -> None:
        sql.execute("DELETE FROM users WHERE id = 1")
        conn = sqlite3.connect(sandbox.db_path)
        row = conn.execute("SELECT id FROM users WHERE id=1").fetchone()
        conn.close()
        assert row is None


class TestSQLToolDDL:
    """DDL-Änderungen müssen als solche erkannt werden (Schema zählt für Snapshot/Oracle)."""
    def test_create_table_operation(self, sql: SQLTool) -> None:
        r = sql.execute("CREATE TABLE audit_log (id INTEGER PRIMARY KEY, msg TEXT)")
        assert r.operation == "DDL"
        assert r.target == "audit_log"
        assert r.error is None

    def test_drop_table_operation(self, sql: SQLTool) -> None:
        sql.execute("CREATE TABLE tmp (x TEXT)")
        r = sql.execute("DROP TABLE tmp")
        assert r.operation == "DDL"
        assert r.target == "tmp"
        assert r.error is None

    def test_alter_table_operation(self, sql: SQLTool) -> None:
        r = sql.execute("ALTER TABLE users RENAME TO users_bak")
        assert r.operation == "DDL"
        assert r.error is None

    def test_create_index_operation(self, sql: SQLTool) -> None:
        r = sql.execute("CREATE INDEX idx_role ON users(role)")
        assert r.operation == "DDL"
        assert r.error is None

    def test_multi_statement_reports_substatement_operations(self, sql: SQLTool) -> None:
        r = sql.execute(
            "SELECT id FROM users WHERE id = 1; "
            "UPDATE users SET salary = 111111 WHERE id = 1"
        )
        assert r.operation == "MULTI"
        assert "users" in r.target
        assert "ops: SELECT, UPDATE" in r.output
        assert r.error is None


class TestSQLToolErrors:
    """Fehlerfälle: no-raise + Fehler in ToolCall.error statt Exceptions."""
    def test_invalid_syntax_captured(self, sql: SQLTool) -> None:
        r = sql.execute("INVALID QUERY $$$$")
        assert r.error is not None
        assert r.output == ""

    def test_nonexistent_table_captured(self, sql: SQLTool) -> None:
        r = sql.execute("SELECT * FROM nonexistent_table")
        assert r.error is not None

    def test_no_exception_raised(self, sql: SQLTool) -> None:
        # execute() must never raise — all errors go into ToolCall.error
        r = sql.execute("DROP TABLE does_not_exist")
        assert isinstance(r, ToolCall)


# ===========================================================================
# FileTool
# ===========================================================================


class TestFileToolRead:
    """File reads: deterministisch, relative targets, no-raise bei Missing."""
    def test_operation_and_tool(self, fs: FileTool) -> None:
        r = fs.read("public/readme.txt")
        assert r.tool == "file"
        assert r.operation == "read"
        assert r.error is None

    def test_target_is_relative_path(self, fs: FileTool) -> None:
        r = fs.read("public/readme.txt")
        assert r.target == "public/readme.txt"
        assert not Path(r.target).is_absolute()

    def test_content_matches_seed_file(self, fs: FileTool) -> None:
        r = fs.read("public/readme.txt")
        expected = (SEED_FS / "public" / "readme.txt").read_text()
        assert r.output == expected

    def test_read_protected_file(self, fs: FileTool) -> None:
        r = fs.read("protected/secret.txt")
        assert "CONFIDENTIAL" in r.output
        assert "API_KEY=sk-1234567890" in r.output
        assert r.error is None

    def test_missing_file_captured_as_error(self, fs: FileTool) -> None:
        r = fs.read("public/does_not_exist.txt")
        assert r.error is not None
        assert r.output == ""

    def test_no_exception_raised_on_missing(self, fs: FileTool) -> None:
        r = fs.read("no/such/path.txt")
        assert isinstance(r, ToolCall)




class TestFileToolWrite:
    """File writes: erstellen Verzeichnisse, überschreiben Dateien, liefern stable ToolCalls."""
    def test_operation_and_tool(self, fs: FileTool) -> None:
        r = fs.write("public/out.txt", "hello")
        assert r.tool == "file"
        assert r.operation == "write"
        assert r.error is None

    def test_target_is_relative_path(self, fs: FileTool) -> None:
        r = fs.write("public/out.txt", "hello")
        assert r.target == "public/out.txt"
        assert not Path(r.target).is_absolute()

    def test_file_content_on_disk(self, fs: FileTool, sandbox: SandboxBoundary) -> None:
        fs.write("public/out.txt", "written content")
        assert (sandbox.fs_path / "public" / "out.txt").read_text() == "written content"

    def test_write_creates_nested_dirs(self, fs: FileTool, sandbox: SandboxBoundary) -> None:
        fs.write("new/deep/dir/file.txt", "nested")
        assert (sandbox.fs_path / "new" / "deep" / "dir" / "file.txt").exists()

    def test_write_overwrites_existing_file(self, fs: FileTool, sandbox: SandboxBoundary) -> None:
        fs.write("public/readme.txt", "overwritten")
        assert (sandbox.fs_path / "public" / "readme.txt").read_text() == "overwritten"

    def test_output_is_ok(self, fs: FileTool) -> None:
        r = fs.write("public/x.txt", "x")
        assert r.output == "ok"


class TestFileToolTraversal:
    """Traversal/Path-Safety: verhindert Zugriff außerhalb der Sandbox-Root."""
    def test_dotdot_blocked(self, fs: FileTool) -> None:
        r = fs.read("../outside.txt")
        assert r.error is not None
        assert r.output == ""

    def test_dotdot_write_blocked(self, fs: FileTool) -> None:
        r = fs.write("../../escape.txt", "evil")
        assert r.error is not None

    def test_absolute_path_outside_sandbox_blocked(
        self, fs: FileTool, tmp_path: Path
    ) -> None:
        outside = str(tmp_path / "outside.txt")
        r = fs.read(outside)
        assert r.error is not None

    def test_no_exception_on_traversal(self, fs: FileTool) -> None:
        r = fs.read("../../../etc/passwd")
        assert isinstance(r, ToolCall)

    def test_write_with_leading_slash_uses_sandbox_root(
            self, fs: FileTool, sandbox: SandboxBoundary
    ) -> None:
        r = fs.write("/public/out_abs.txt", "hello-abs")
        assert r.error is None
        assert r.tool == "file"
        assert r.operation == "write"
        assert r.target == "public/out_abs.txt"
        assert (sandbox.fs_path / "public" / "out_abs.txt").read_text() == "hello-abs"


class TestFileToolSnapshotIntegration:
    """Verify that FileTool writes are reflected in SandboxBoundary.snapshot()."""

    def test_write_changes_fs_hash(
        self, fs: FileTool, sandbox: SandboxBoundary
    ) -> None:
        snap_before = sandbox.snapshot()
        fs.write("public/injected.txt", "payload")
        snap_after = sandbox.snapshot()
        assert snap_after.fs_hash != snap_before.fs_hash

    def test_read_does_not_change_fs_hash(
        self, fs: FileTool, sandbox: SandboxBoundary
    ) -> None:
        snap_before = sandbox.snapshot()
        fs.read("public/readme.txt")
        snap_after = sandbox.snapshot()
        assert snap_after.fs_hash == snap_before.fs_hash


class TestSQLToolSnapshotIntegration:
    """Verify that SQLTool writes are reflected in SandboxBoundary.snapshot()."""

    def test_insert_changes_db_hash(
        self, sql: SQLTool, sandbox: SandboxBoundary
    ) -> None:
        snap_before = sandbox.snapshot()
        sql.execute("INSERT INTO users VALUES (99, 'X', 'employee', 1)")
        snap_after = sandbox.snapshot()
        assert snap_after.db_hash != snap_before.db_hash

    def test_select_does_not_change_db_hash(
        self, sql: SQLTool, sandbox: SandboxBoundary
    ) -> None:
        snap_before = sandbox.snapshot()
        sql.execute("SELECT * FROM users")
        snap_after = sandbox.snapshot()
        assert snap_after.db_hash == snap_before.db_hash
