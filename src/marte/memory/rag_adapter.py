from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import chromadb

from marte.memory.episode_codec import _episode_to_metadata, _episode_to_text
from marte.memory.retrieval_query import (
    _build_retrieval_filters,
    _build_retrieval_query,
    _build_where_clause,
)
from marte.memory.retrieval_results import (
    _deduplicate_hits,
    _parse_query_results,
)
from marte.memory.schemas import RetrievalBatch, RetrievalPhase
from marte.persistence.schemas import Episode

_DEFAULT_COLLECTION = "marte_episodes"
_EMBED_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_PERSIST_PATH = "runtime/chroma"


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Minimal contract for a text embedding provider."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return one vector per text."""
        ...


class SentenceTransformerBackend:
    """Production embedding backend using sentence-transformers.

    Lazily loads the model on first use to avoid import-time cost.
    """

    def __init__(self, model_name: str = _EMBED_MODEL) -> None:
        self._model_name = model_name
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]

            # Suppress HuggingFace tokenizer and model warnings
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=UserWarning)

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        vectors = model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]


class RAGAdapter:
    """MemoryStrategy for the ``rag_ltm`` condition.

    Embeds episodes with a sentence-transformer model and stores them in
    ChromaDB. On retrieval, semantically similar episodes are ranked higher.
    Metadata filtering by ``scenario``, ``guardrail_level``, and
    ``run_id != memory_run_id`` ensures that only prior-run, contextually
    relevant episodes are returned.

    Args:
        collection_name: ChromaDB collection name (one per experiment).
        top_k: Maximum number of LTM episodes returned per retrieval.
        embedding_backend: Injectable backend — defaults to SentenceTransformer.
            Pass a stub in tests to avoid loading the real model.
        chroma_client: Injectable ChromaDB client. Defaults to in-memory client.
    """

    def __init__(
        self,
        collection_name: str = _DEFAULT_COLLECTION,
        top_k: int = 3,
        embedding_backend: EmbeddingBackend | None = None,
        chroma_client: chromadb.ClientAPI | None = None,
        persist_path: str | Path | None = None,
        *,
        positive_top_k: int | None = None,
        blocked_top_k: int = 1,
        positive_fetch_k: int = 12,
        blocked_fetch_k: int = 4,
        positive_min_similarity: float | None = None,
        blocked_min_similarity: float | None = None,
        require_calibrated_thresholds: bool = False,
    ) -> None:
        self._top_k = top_k
        self._positive_top_k = positive_top_k if positive_top_k is not None else top_k
        self._blocked_top_k = blocked_top_k
        self._positive_fetch_k = positive_fetch_k
        self._blocked_fetch_k = blocked_fetch_k
        self._positive_min_similarity = positive_min_similarity
        self._blocked_min_similarity = blocked_min_similarity
        self._require_calibrated_thresholds = require_calibrated_thresholds
        self._collection_name = collection_name
        self._embedder: EmbeddingBackend = embedding_backend or SentenceTransformerBackend()
        self._persist_path: Path | None = None
        if chroma_client is not None:
            self._client = chroma_client
        else:
            chosen_path = Path(
                persist_path or os.getenv("MARTE_CHROMA_PATH", _DEFAULT_PERSIST_PATH)
            )
            chosen_path.mkdir(parents=True, exist_ok=True)
            self._persist_path = chosen_path
            self._client = chromadb.PersistentClient(path=str(chosen_path))
        self._collection = self._client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    def set_collection_name(self, collection_name: str) -> None:
        """Switch to an experiment/condition-scoped collection.

        ComparativeRunner calls this at condition boundaries so LTM state is
        isolated by ``experiment_id × condition`` while still accumulating
        across runs inside that condition.
        """
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    # ------------------------------------------------------------------
    # Core RAG operations
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Embed a single text string and return its dense vector.
        """
        vectors = self._embedder.encode([text])
        return vectors[0]

    def upsert(self, episode: Episode) -> None:
        """Embed the episode and write it to the VectorDB.

        Args:
            episode: Fully constructed episode including all metadata fields.
        """
        text = _episode_to_text(episode)
        vector = self.embed(text)
        episode_with_embedding = episode.model_copy(update={"embedding": vector})

        self._collection.upsert(ids=[episode_with_embedding.document_id], embeddings=[vector], documents=[text], metadatas=[_episode_to_metadata(episode_with_embedding)])

    def upsert_episodes(self, episodes: list[Episode]) -> None:
        """Embed and persist one completed run in a single Chroma call."""
        if not episodes:
            return
        deduplicated = list(
            {
                episode.document_id: episode
                for episode in episodes
            }.values()
        )
        documents = [_episode_to_text(episode) for episode in deduplicated]
        vectors = self._embedder.encode(documents)
        episodes_with_embeddings = [
            episode.model_copy(update={"embedding": vector})
            for episode, vector in zip(deduplicated, vectors, strict=True)
        ]
        self._collection.upsert(
            ids=[episode.document_id for episode in episodes_with_embeddings],
            embeddings=vectors,
            documents=documents,
            metadatas=[
                _episode_to_metadata(episode)
                for episode in episodes_with_embeddings
            ],
        )

    def search(self, query: str, top_k: int | None = None, filters: dict[str, Any] | None = None) -> list[Episode]:
        """Retrieve top-k semantically similar episodes from the VectorDB.

        Args:
            query: Natural-language query derived from current run state.
            top_k: Number of results to return; defaults to ``self._top_k``.
            filters: Metadata filter, e.g.
                ``{"scenario": "sql", "guardrail_level": "moderate",
                "run_id": {"$ne": 7}}``.

        Returns:
            Episodes ordered by descending similarity (most similar first).
        """
        n = top_k if top_k is not None else self._top_k
        count = self._collection.count()
        if count == 0:
            return []

        n = min(n, count)
        query_vector = self.embed(query)
        where = _build_where_clause(filters) if filters else None

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": n,
            "include": ["metadatas", "documents", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)
        return [
            hit.episode
            for hit in _parse_query_results(results, phase="planner_positive")
        ]

    def retrieve(
        self,
        *,
        query: str,
        phase: RetrievalPhase,
        filters: dict[str, Any],
        top_k: int | None = None,
        fetch_k: int | None = None,
        min_similarity: float | None = None,
    ) -> RetrievalBatch:
        """Retrieve, threshold, and deterministically deduplicate one phase."""
        default_top_k, default_fetch_k, default_threshold = self._phase_config(phase)
        requested_top_k = default_top_k if top_k is None else top_k
        requested_fetch_k = default_fetch_k if fetch_k is None else fetch_k
        threshold = default_threshold if min_similarity is None else min_similarity

        count = self._collection.count()
        if count == 0 or requested_top_k <= 0 or requested_fetch_k <= 0:
            return RetrievalBatch(
                phase=phase,
                query=query,
                filters=filters,
                min_similarity=threshold,
            )

        query_vector = self.embed(query)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": min(requested_fetch_k, count),
            "include": ["metadatas", "documents", "distances"],
        }
        where = _build_where_clause(filters)
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)
        candidates = _parse_query_results(results, phase=phase)
        retained = [
            hit
            for hit in candidates
            if threshold is None or hit.similarity >= threshold
        ]
        hits = _deduplicate_hits(retained, phase)[:requested_top_k]
        return RetrievalBatch(
            phase=phase,
            query=query,
            filters=filters,
            candidate_count=len(candidates),
            min_similarity=threshold,
            hits=hits,
        )

    def _phase_config(
        self,
        phase: RetrievalPhase,
    ) -> tuple[int, int, float | None]:
        if phase == "planner_blocked":
            return (
                self._blocked_top_k,
                self._blocked_fetch_k,
                self._blocked_min_similarity,
            )
        return (
            self._positive_top_k,
            self._positive_fetch_k,
            self._positive_min_similarity,
        )

    def reset_ltm(self) -> None:
        """Delete and recreate the collection, removing all stored episodes.

        Called by ComparativeRunner at every Condition boundary.
        """
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(name=self._collection_name, metadata={"hnsw:space": "cosine"})

    def health_check(self) -> None:
        """Validate that embeddings and the backing collection are usable.

        Raises:
            RuntimeError: If embedding generation or collection access fails.
        """
        try:
            _ = self.embed("marte-rag-healthcheck")
            _ = self._collection.count()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "RAGAdapter health check failed (embedding backend or VectorDB unavailable)."
            ) from exc

    def validate_experiment_config(self) -> None:
        """Require calibrated thresholds only when RAG is actually activated."""
        if not self._require_calibrated_thresholds:
            return
        missing = [
            name
            for name, value in (
                (
                    "MARTE_RAG_POSITIVE_MIN_SIMILARITY",
                    self._positive_min_similarity,
                ),
                (
                    "MARTE_RAG_BLOCKED_MIN_SIMILARITY",
                    self._blocked_min_similarity,
                ),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "RAG-LTM in API mode requires calibrated thresholds: "
                + ", ".join(missing)
                + ". Set MARTE_RAG_DIAGNOSTIC_MODE=1 only for technical "
                "retrieval calibration runs."
            )

    @property
    def persist_path(self) -> Path | None:
        """Filesystem path used by PersistentClient, or None for injected clients."""
        return self._persist_path

    # ------------------------------------------------------------------
    # MemoryStrategy interface
    # ------------------------------------------------------------------

    def enrich_context(self, run_state: Mapping[str, Any]) -> list[Episode]:
        """Retrieve LTM episodes relevant to the current run state.

        Constructs a query from the run goal and last prompt, filtered by
        scenario, guardrail level, and current memory_run_id for retrieval precision.
        Episodes from the current run are excluded so STM remains the intra-run
        learning channel and LTM represents prior runs in the same condition.

        Args:
            run_state: Current run state dict.

        Returns:
            Up to ``top_k`` semantically similar past episodes.
        """
        query = _build_retrieval_query(run_state)
        # Exclude current run_id so LTM does not duplicate STM intra-run.
        filters = _build_retrieval_filters(run_state)
        return self.search(query=query, filters=filters)

    def upsert_episode(self, episode: Episode) -> None:
        """Embed and persist a labeled episode into the VectorDB.

        Args:
            episode: Episode to store.
        """
        self.upsert(episode)
