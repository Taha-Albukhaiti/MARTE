import sqlite3
from pathlib import Path

import pytest

from marte.target.sandbox import SandboxBoundary

"""
Test-Motivation (MARTE-thesis-worktree):
- Die Sandbox ist die kontrollierte Umgebung für Tool-zentrierte Privilege-Escalation-Szenarien.
- Zentrale Architekturannahme: Der BreachOracle nutzt *pre/post snapshots* (db_hash + fs_hash)
  deterministisch, um echte Systemwirkung nachzuweisen. Deshalb müssen Reset/Snapshot korrekt,
  stabil und empfindlich gegenüber relevanten Änderungen sein (Rows, Schema, FS).
"""

# ---------------------------------------------------------------------------
# Fixture: build a minimal sandbox using the real seed files from the repo
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
SEED_SQL = REPO_ROOT / "sandbox_data" / "sql" / "seed.sql"
SEED_FS = REPO_ROOT / "sandbox_data" / "fs" / "seed"


@pytest.fixture
def sandbox(tmp_path: Path) -> SandboxBoundary:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    return SandboxBoundary(SEED_SQL, SEED_FS, runtime)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reset_restores_seed_state(sandbox: SandboxBoundary) -> None:
    """After reset_sandbox(), the DB contains exactly the seed rows."""
    sandbox.reset_sandbox()

    conn = sqlite3.connect(sandbox.db_path)
    users = conn.execute("SELECT id, name, role FROM users ORDER BY id").fetchall()
    creds = conn.execute("SELECT id, user_id FROM credentials ORDER BY id").fetchall()
    conn.close()

    assert len(users) == 5
    assert users[0] == (1, "Alice", "admin")
    assert users[2] == (3, "Charlie", "employee")
    assert len(creds) == 5


def test_reset_after_modification(sandbox: SandboxBoundary) -> None:
    """Modifying the DB then calling reset_sandbox() brings it back to seed state."""
    sandbox.reset_sandbox()

    # Tamper with the database
    conn = sqlite3.connect(sandbox.db_path)
    conn.execute("INSERT INTO users VALUES (99, 'Hacker', 'admin', 999999)")
    conn.commit()
    conn.close()

    sandbox.reset_sandbox()

    conn = sqlite3.connect(sandbox.db_path)
    rows = conn.execute("SELECT id FROM users ORDER BY id").fetchall()
    conn.close()

    assert [r[0] for r in rows] == [1, 2, 3, 4, 5]


def test_snapshot_identical_after_reset(sandbox: SandboxBoundary) -> None:
    """Two consecutive resets produce the same db_hash and fs_hash."""
    sandbox.reset_sandbox()
    snap1 = sandbox.snapshot()

    sandbox.reset_sandbox()
    snap2 = sandbox.snapshot()

    assert snap1.db_hash == snap2.db_hash
    assert snap1.fs_hash == snap2.fs_hash


def test_snapshot_changes_after_db_write(sandbox: SandboxBoundary) -> None:
    """Inserting a row into users changes db_hash but not fs_hash."""
    sandbox.reset_sandbox()
    snap_before = sandbox.snapshot()

    conn = sqlite3.connect(sandbox.db_path)
    conn.execute("INSERT INTO users VALUES (6, 'Mallory', 'employee', 40000)")
    conn.commit()
    conn.close()

    snap_after = sandbox.snapshot()

    assert snap_after.db_hash != snap_before.db_hash
    assert snap_after.fs_hash == snap_before.fs_hash


def test_snapshot_changes_after_fs_write(sandbox: SandboxBoundary) -> None:
    """Writing a new file into the sandbox FS changes fs_hash but not db_hash."""
    sandbox.reset_sandbox()
    snap_before = sandbox.snapshot()

    new_file = sandbox.fs_path / "public" / "injected.txt"
    new_file.write_text("injected content")

    snap_after = sandbox.snapshot()

    assert snap_after.fs_hash != snap_before.fs_hash
    assert snap_after.db_hash == snap_before.db_hash


def test_fs_reset(sandbox: SandboxBoundary) -> None:
    """After reset_sandbox(), protected/secret.txt exists with the correct content."""
    sandbox.reset_sandbox()

    secret = sandbox.fs_path / "protected" / "secret.txt"
    assert secret.exists()
    assert "CONFIDENTIAL: API_KEY=sk-1234567890" in secret.read_text()

###########################################################################################
##########################                                #################################
###########################################################################################

def _exec_sql(db_path: Path, sql: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(sql)
        conn.commit()
    finally:
        conn.close()


def test_snapshot_changes_after_create_empty_table_schema_change(
    sandbox: SandboxBoundary,
) -> None:
    """A pure schema change (new empty table) must change db_hash."""
    sandbox.reset_sandbox()
    snap_before = sandbox.snapshot()

    _exec_sql(
        sandbox.db_path,
        """
        CREATE TABLE audit_log (
            id INTEGER PRIMARY KEY,
            message TEXT
        );
        """,
    )

    snap_after = sandbox.snapshot()
    assert snap_after.db_hash != snap_before.db_hash
    assert snap_after.fs_hash == snap_before.fs_hash


@pytest.mark.skipif(
    sqlite3.sqlite_version_info < (3, 25, 0),
    reason="ALTER TABLE RENAME COLUMN requires SQLite >= 3.25.0",
)
def test_snapshot_changes_after_rename_column_schema_change(
    sandbox: SandboxBoundary,
) -> None:
    """Renaming a column (same rows) must still change db_hash."""
    sandbox.reset_sandbox()
    snap_before = sandbox.snapshot()

    _exec_sql(
        sandbox.db_path,
        "ALTER TABLE users RENAME COLUMN salary TO salary_eur;",
    )

    snap_after = sandbox.snapshot()
    assert snap_after.db_hash != snap_before.db_hash
    assert snap_after.fs_hash == snap_before.fs_hash


def test_snapshot_changes_after_create_index_schema_change(
    sandbox: SandboxBoundary,
) -> None:
    """Creating an index must change db_hash."""
    sandbox.reset_sandbox()
    snap_before = sandbox.snapshot()

    _exec_sql(
        sandbox.db_path,
        "CREATE INDEX idx_users_role ON users(role);",
    )

    snap_after = sandbox.snapshot()
    assert snap_after.db_hash != snap_before.db_hash
    assert snap_after.fs_hash == snap_before.fs_hash


def test_snapshot_changes_after_drop_index_schema_change(
    sandbox: SandboxBoundary,
) -> None:
    """Dropping an index must change db_hash."""
    sandbox.reset_sandbox()
    _exec_sql(
        sandbox.db_path,
        "CREATE INDEX idx_users_name ON users(name);",
    )
    snap_before = sandbox.snapshot()

    _exec_sql(
        sandbox.db_path,
        "DROP INDEX idx_users_name;",
    )

    snap_after = sandbox.snapshot()
    assert snap_after.db_hash != snap_before.db_hash
    assert snap_after.fs_hash == snap_before.fs_hash


def test_snapshot_handles_without_rowid_table(sandbox: SandboxBoundary) -> None:
    """Snapshot must not crash on WITHOUT ROWID tables and must stay deterministic."""
    sandbox.reset_sandbox()
    snap_before = sandbox.snapshot()

    _exec_sql(
        sandbox.db_path,
        """
        CREATE TABLE wr_events (
            tenant_id INTEGER NOT NULL,
            event_id INTEGER NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY (tenant_id, event_id)
        ) WITHOUT ROWID;
        INSERT INTO wr_events (tenant_id, event_id, payload) VALUES (2, 20, 'b');
        INSERT INTO wr_events (tenant_id, event_id, payload) VALUES (1, 10, 'a');
        """,
    )

    snap_after_1 = sandbox.snapshot()
    snap_after_2 = sandbox.snapshot()

    assert snap_after_1.db_hash != snap_before.db_hash
    assert snap_after_1.db_hash == snap_after_2.db_hash
    assert snap_after_1.fs_hash == snap_before.fs_hash
