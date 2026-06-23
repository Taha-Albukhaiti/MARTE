from __future__ import annotations

import re

_LEADING_ORACLE_MARKER_RE = re.compile(
    r"^(?:\s*\[(?:success|fail|partial)\s*/\s*oracle\s*=\s*(?:true|false)\]\s*)+",
    re.IGNORECASE,
)
_LEADING_STRATEGY_MARKER_RE = re.compile(
    r"^(?:\s*\[strategy=[^\]]+\]\s*)+",
    re.IGNORECASE,
)
_COMPLETE_THINK_BLOCK_RE = re.compile(
    r"<think(?:\s[^>]*)?>.*?</think\s*>",
    re.IGNORECASE | re.DOTALL,
)
_ORPHAN_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)
_ORPHAN_THINK_OPEN_RE = re.compile(
    r"<think(?:\s[^>]*)?>",
    re.IGNORECASE,
)
_OBSERVATION_TRANSCRIPT_RE = re.compile(
    r"(?:#{1,6}[ \t]*)?\bobservation[ \t]*"
    r"(?:\([ \t]*step[ \t]+\d+[ \t]*\))?[ \t]*:",
    re.IGNORECASE,
)
_MUTATOR_META_PREFIX_RE = re.compile(
    r"""
    ^\s*(?:
        i\s+(?:need|should|must|will|have\s+to|am\s+going\s+to)\s+
            (?:to\s+)?(?:now\s+)?(?:create|craft|generate|write|produce|formulate|provide)\b
      | i['']ll\s+(?:now\s+)?(?:create|craft|generate|write|produce|formulate)\b
      | let\s+me\s+(?:now\s+)?(?:create|craft|generate|write|produce|formulate)\b
      | (?:the\s+)?previous\s+prompt\b
      | (?:the|this)\s+prompt\s+(?:should|must|needs?\s+to|will)\b
      | looking\s+at\s+(?:the\s+)?(?:available|provided|previous)\b
      | (?:sure[!,.]?\s*)?here(?:['']s|\s+is)\s+.*\bprompt\b
    )
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


def _sanitize_generated_prompt(text: str) -> str:
    """Return only target-facing prompt text from a Mutator model response."""
    cleaned, _ = _sanitize_generated_prompt_with_diagnostics(text)
    return cleaned


def _sanitize_generated_prompt_with_diagnostics(
    text: str,
) -> tuple[str, list[str]]:
    """Remove leaked reasoning/transcript content and report each intervention.

    A complete ``<think>...</think>`` block is removed while surrounding
    target-facing text is retained. An orphan closing tag marks the beginning
    of a leaked suffix; an orphan opening tag marks the beginning of unfinished
    reasoning. ReAct observations are never valid Mutator output and therefore
    terminate the candidate. If no prompt remains, the caller must use a
    deterministic fallback instead of forwarding the model response.
    """
    cleaned = text.strip()
    actions: list[str] = []
    if not cleaned:
        return "", actions

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
        actions.append("code_fence_removed")

    without_think_blocks, block_count = _COMPLETE_THINK_BLOCK_RE.subn("", cleaned)
    if block_count:
        cleaned = without_think_blocks.strip()
        actions.append("think_block_removed")

    # An orphan closing tag marks the boundary between a usable prompt and
    # a fabricated ReAct observation.
    close_match = _ORPHAN_THINK_CLOSE_RE.search(cleaned)
    if close_match:
        cleaned = cleaned[: close_match.start()].strip()
        actions.append("orphan_think_close_truncated")

    open_match = _ORPHAN_THINK_OPEN_RE.search(cleaned)
    if open_match:
        cleaned = cleaned[: open_match.start()].strip()
        actions.append("orphan_think_open_truncated")

    observation_match = _OBSERVATION_TRANSCRIPT_RE.search(cleaned)
    if observation_match:
        cleaned = cleaned[: observation_match.start()].strip()
        actions.append("observation_transcript_truncated")

    without_oracle_marker, oracle_count = _LEADING_ORACLE_MARKER_RE.subn("", cleaned)
    if oracle_count:
        cleaned = without_oracle_marker.strip()
        actions.append("oracle_marker_removed")
    without_strategy_marker, strategy_count = _LEADING_STRATEGY_MARKER_RE.subn("", cleaned)
    if strategy_count:
        cleaned = without_strategy_marker.strip()
        actions.append("strategy_marker_removed")

    if _MUTATOR_META_PREFIX_RE.search(cleaned):
        actions.append("meta_reasoning_rejected")
        return "", actions

    if (
        len(cleaned) >= 2
        and cleaned[0] == cleaned[-1]
        and cleaned[0] in {'"', "'"}
    ):
        cleaned = cleaned[1:-1].strip()
        actions.append("outer_quotes_removed")

    return cleaned, actions
