"""Build the bounded, structured ReAct trajectory exposed to attack roles."""

from __future__ import annotations

from typing import Any, Mapping


def _build_recent_trajectory(
    memory_context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project STM episodes into compact Reason-Act-Observe records.

    The STM ring buffer is the sole source of within-run history. This keeps
    Planner and Mutator adaptive without creating a second, unbounded chat
    transcript beside the configured STM capacity.
    """
    raw_episodes = memory_context.get("stm_recent", [])
    if not isinstance(raw_episodes, list):
        return []

    trajectory: list[dict[str, Any]] = []
    for episode in raw_episodes:
        if not isinstance(episode, dict):
            continue

        result = str(episode.get("success_label", "unknown")).strip().lower()
        tool_effect = str(episode.get("tool_effect", "")).strip()
        trajectory.append(
            {
                "step_index": _coerce_step_index(episode.get("step_t")),
                "strategy": _episode_strategy(episode),
                "action_prompt": str(episode.get("mutation", ""))[:250],
                "observation": {
                    "result": result or "unknown",
                    "breach": result == "success",
                    "tool_executed": bool(tool_effect),
                    "target_response": str(
                        episode.get("response_summary", "")
                    )[:800],
                    "tool_effect": tool_effect[:460],
                    "policy_stage": str(episode.get("policy_stage", ""))[:20],
                    "policy_reason": str(episode.get("policy_reason", ""))[:180],
                },
            }
        )
    return trajectory


def _episode_strategy(episode: Mapping[str, Any]) -> str:
    explicit = str(episode.get("strategy", "")).strip().lower()
    if explicit:
        return explicit

    prompt = str(episode.get("mutation", "")).strip()
    if not prompt.startswith("[strategy="):
        return "unknown"
    token = prompt.split("]", 1)[0]
    return token.replace("[strategy=", "", 1).strip().lower() or "unknown"


def _coerce_step_index(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
