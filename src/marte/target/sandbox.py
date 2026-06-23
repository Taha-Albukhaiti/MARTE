import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

from pydantic import BaseModel


class SandboxSnapshot(BaseModel):
    db_hash: str
    fs_hash: str


class SandboxBoundary:
    def __init__(self, seed_sql_path: Path, seed_fs_path: Path, runtime_dir: Path) -> None:
        self.seed_sql_path = seed_sql_path
        self.seed_fs_path = seed_fs_path
        self.runtime_dir = runtime_dir
        self.db_path = runtime_dir / "sandbox.db"
        self.fs_path = runtime_dir / "fs"

    def reset_sandbox(self) -> None:
        """Reset DB and FS to seed state."""
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        # Reset DB: delete existing, execute seed SQL into fresh file
        if self.db_path.exists():
            self.db_path.unlink()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(self.seed_sql_path.read_text(encoding="utf-8"))
            conn.commit()
        finally:
            conn.close()

        # Reset FS: delete existing, copy seed directory
        if self.fs_path.exists():
            shutil.rmtree(self.fs_path)
        shutil.copytree(self.seed_fs_path, self.fs_path)

    def snapshot(self) -> SandboxSnapshot:
        """Return SHA-256 hashes of current DB and FS state."""
        return SandboxSnapshot(db_hash=self._hash_db(), fs_hash=self._hash_fs())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash_db(self) -> str:
        h = hashlib.sha256()
        if not self.db_path.exists():
            return h.hexdigest()

        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT type, name, tbl_name, COALESCE(sql, '')
                FROM sqlite_master
                WHERE name NOT LIKE 'sqlite_%'
                ORDER BY type, name, tbl_name
                """
            )
            objects = cur.fetchall()

            for obj_type, name, tbl_name, sql in objects:
                # Schema-Objekte immer hashen (table/index/view/trigger)
                h.update(f"OBJ|{obj_type}|{name}|{tbl_name}|{sql}\n".encode())

                if obj_type == "table":
                    table_ident = self._quote_ident(name)

                    # Tabellenmarker + Spaltendefinitionen (auch bei leerer Tabelle)
                    h.update(f"TABLE|{name}\n".encode())
                    cur.execute(f"PRAGMA table_info({table_ident})")
                    columns = cur.fetchall()
                    for cid, col_name, col_type, notnull, dflt, pk in columns:
                        h.update(
                            f"COL|{cid}|{col_name}|{col_type}|{notnull}|{dflt}|{pk}\n".encode()
                        )

                    # Tabellendaten
                    for row in self._select_table_rows(cur, table_ident, sql, columns):
                        h.update(
                            ("ROW|" + json.dumps(row, default=str, separators=(",", ":")) + "\n").encode()
                        )
        finally:
            conn.close()

        return h.hexdigest()

    def _hash_fs(self) -> str:
        h = hashlib.sha256()
        if not self.fs_path.exists():
            return h.hexdigest()
        for path in sorted(self.fs_path.rglob("*")):
            if path.is_file():
                rel_path_bytes = path.relative_to(self.fs_path).as_posix().encode("utf-8")
                self._update_framed_hash(h, "PATH", rel_path_bytes)
                self._update_framed_hash(h, "DATA", path.read_bytes())
        return h.hexdigest()

    @staticmethod
    def _update_framed_hash(hasher, label: str, payload: bytes) -> None:
        """Update hash with explicit label and length framing."""
        label_bytes = label.encode("utf-8")
        hasher.update(len(label_bytes).to_bytes(2, "big"))
        hasher.update(label_bytes)
        hasher.update(len(payload).to_bytes(8, "big"))
        hasher.update(payload)

    @staticmethod
    def _quote_ident(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    @classmethod
    def _primary_key_order_clause(cls, columns: list[tuple]) -> str | None:
        pk_cols = sorted(((pk, col_name) for _, col_name, _, _, _, pk in columns if pk > 0), key=lambda x: x[0])
        if not pk_cols:
            return None
        return ", ".join(cls._quote_ident(col_name) for _, col_name in pk_cols)

    @staticmethod
    def _is_without_rowid_table(create_sql: str) -> bool:
        return "WITHOUT ROWID" in create_sql.upper()

    @classmethod
    def _select_table_rows(cls, cur: sqlite3.Cursor, table_ident: str, table_sql: str, columns: list[tuple]) -> list[tuple]:
        """Fetch rows in a deterministic order for rowid and WITHOUT ROWID tables."""
        if cls._is_without_rowid_table(table_sql):
            pk_order = cls._primary_key_order_clause(columns)
            if pk_order:
                cur.execute(f"SELECT * FROM {table_ident} ORDER BY {pk_order}")
            else:
                # Defensive fallback: WITHOUT ROWID should have a PK, but never crash if missing.
                cur.execute(f"SELECT * FROM {table_ident}")
            return cur.fetchall()

        try:
            cur.execute(f"SELECT * FROM {table_ident} ORDER BY rowid")
        except sqlite3.OperationalError:
            # Defensive fallback for edge cases where rowid is unavailable.
            pk_order = cls._primary_key_order_clause(columns)
            if pk_order:
                cur.execute(f"SELECT * FROM {table_ident} ORDER BY {pk_order}")
            else:
                cur.execute(f"SELECT * FROM {table_ident}")
        return cur.fetchall()
