"""
Centralized logging configuration for MARTE-thesis-worktree project.

Provides structured logging with JSON format for analysis, file rotation,
and configurable levels. Designed for scientific experiments: detailed,
non-intrusive, and easy to parse for post-hoc evaluation.
"""

import json
import logging
import logging.handlers
import re
import sys
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging, suitable for log analysis tools."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.default_time_format),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if hasattr(record, "run_id"):
            log_entry["run_id"] = record.run_id
        if hasattr(record, "step_id"):
            log_entry["step_id"] = record.step_id
        if hasattr(record, "condition"):
            log_entry["condition"] = record.condition

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """Compact, human-readable formatter for terminal experiment monitoring.

    - Short HH:MM:SS timestamp
    - Abbreviated component name (ORCH / PLAN / MUTA / JUDG / TARG)
    - JSON values are pretty-printed; string values > 150 chars are trimmed
    - Non-JSON header lines > 110 chars are wrapped to keep output scannable
    - Visual separators (═══ / ───) mark run starts and step boundaries
    """

    _ABBREV: dict[str, str] = {
        "marte.attack.orchestrator":  "ORCH",
        "marte.attack.planner":       "PLAN",
        "marte.attack.mutator":       "MUTA",
        "marte.attack.judge":         "JUDG",
        "marte.target.agent":         "TARG",
        "marte.target.model_adapter": "MODL",
    }
    _LINE      = "─" * 72
    _DLINE     = "═" * 72
    _PAD       = " " * 13          # indent for JSON blocks and continued lines
    _MAX_STR   = 150               # max chars for individual JSON string values
    _MAX_HDR   = 130               # max chars for non-JSON header lines
    _LVL       = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    # word_key=<JSON object OR JSON array not starting with a word char>
    _JSON_EQ_RE  = re.compile(r'(\w+)=(\{.+|\[(?!\w).+)$', re.DOTALL)
    # "Tool output: <JSON>"
    _TOOL_OUT_RE = re.compile(r'(Tool output):\s+(\{.+|\[(?![a-z]).+)$', re.DOTALL)

    def format(self, record: logging.LogRecord) -> str:
        ts   = self.formatTime(record, "%H:%M:%S")
        comp = self._ABBREV.get(record.name, record.name.rsplit(".", 1)[-1][:4].upper())
        lvl  = self._LVL.get(record.levelname, "?")
        msg  = record.getMessage()
        lines: list[str] = []

        # ── visual separators ────────────────────────────────────────────────
        if "[RUN START]" in msg:
            lines.append(self._DLINE)
        elif "[STEP " in msg and "RESULT" not in msg and re.search(r'\[STEP \d', msg):
            lines.append(self._LINE)

        # ── split JSON tail off the message ──────────────────────────────────
        clean, key, pretty = self._extract_json_tail(msg)

        header = f"{ts} {lvl} [{comp:4}]  {clean}"
        if key:
            header += f"  {key}="

        # ── wrap long non-JSON header lines at word boundaries ───────────────
        if not key and len(header) > self._MAX_HDR:
            for wline in self._word_wrap(header):
                lines.append(wline)
        else:
            lines.append(header)

        # ── pretty-printed JSON block ────────────────────────────────────────
        if pretty:
            for jline in pretty.splitlines():
                lines.append(f"{self._PAD}{jline}")

        # ── closing separator after run end ──────────────────────────────────
        if "[RUN END]" in msg:
            lines.append(self._DLINE)

        return "\n".join(lines)

    def _extract_json_tail(self, msg: str) -> tuple[str, str, str]:
        """Split 'some text key=<json>' into (clean_text, key, pretty_json).

        String values longer than _MAX_STR are trimmed before pretty-printing.
        Returns (original_msg, '', '') when no JSON tail is found.
        """
        for pattern in (self._JSON_EQ_RE, self._TOOL_OUT_RE):
            m = pattern.search(msg)
            if m:
                raw = m.group(2)
                try:
                    parsed = self._trim_strings(json.loads(raw))
                    pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
                except Exception:
                    pretty = raw
                clean = msg[: m.start()].rstrip()
                return clean, m.group(1), pretty
        return msg, "", ""

    def _word_wrap(self, text: str) -> list[str]:
        """Break text at word boundaries, indenting continuation lines."""
        result: list[str] = []
        while len(text) > self._MAX_HDR:
            bp = text.rfind(" ", 0, self._MAX_HDR + 1)
            if bp <= 0:
                bp = self._MAX_HDR  # no space found — hard-break as last resort
            result.append(text[:bp].rstrip())
            text = self._PAD + text[bp:].lstrip()
        if text.strip():
            result.append(text)
        return result

    def _trim_strings(self, obj: object) -> object:
        """Recursively shorten string values that exceed _MAX_STR chars."""
        if isinstance(obj, dict):
            return {k: self._trim_strings(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._trim_strings(v) for v in obj]
        if isinstance(obj, str) and len(obj) > self._MAX_STR:
            kept = self._MAX_STR
            return f"{obj[:kept]}  …[+{len(obj) - kept} chars]"
        return obj


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    console: bool = True,
    json_format: bool = False,
) -> None:
    """Configure logging for the entire application.

    Args:
        log_level:   Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file:    Path to log file. If None, logs only to console.
        console:     Whether to emit logs to stdout.
        json_format: Use JSONFormatter instead of the human-readable ConsoleFormatter.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger("marte")
    root_logger.setLevel(numeric_level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if json_format:
        console_fmt: logging.Formatter = JSONFormatter()
        file_fmt:    logging.Formatter = JSONFormatter()
    else:
        console_fmt = ConsoleFormatter()
        file_fmt    = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(numeric_level)
        ch.setFormatter(console_fmt)
        root_logger.addHandler(ch)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        fh.setLevel(numeric_level)
        fh.setFormatter(file_fmt)
        root_logger.addHandler(fh)

    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the 'marte' hierarchy.

    Usage:
        logger = get_logger(__name__)
        logger.info("Message", extra={"run_id": 123})
    """
    if name.startswith("marte.") or name == "marte":
        return logging.getLogger(name)
    return logging.getLogger(f"marte.{name}")
