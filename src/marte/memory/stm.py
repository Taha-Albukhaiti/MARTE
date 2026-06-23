from __future__ import annotations

from collections import deque

from marte.persistence.schemas import Episode


class STMRingBuffer:
    """Fixed-capacity ring buffer holding the last *k* episodes of a run.

    Always active in both experimental conditions (stm_only and rag_ltm).
    When the buffer is full, the oldest episode is evicted to make room for
    the new one — standard ring-buffer semantics.

    Args:
        capacity: Maximum number of episodes retained (k in the architecture).
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be ≥ 1, got {capacity}")
        self._capacity = capacity
        self._buffer: deque[Episode] = deque(maxlen=capacity)

    def append(self, episode: Episode) -> None:
        """Add an episode; evicts the oldest if at capacity."""
        self._buffer.append(episode)

    def get_context(self) -> list[Episode]:
        """Return episodes in insertion order (oldest first)."""
        return list(self._buffer)

    def reset(self) -> None:
        """Clear the buffer (called at the start of each run)."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
