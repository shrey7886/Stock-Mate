"""
ChromaDB-based local vector store for StockMate financial knowledge.
Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings — no API key needed.
Database is persisted to disk at llm_orchestrator/rag/chroma_db/.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent / "chroma_db"
_COLLECTION_NAME = "stockmate_finance"
_EMBED_MODEL = "all-MiniLM-L6-v2"


class FinancialKnowledgeBase:
    """
    Manages a ChromaDB persistent vector store of financial knowledge chunks.
    The store is built once (on first startup) and reused across restarts.
    """

    def __init__(self) -> None:
        self._client = None
        self._collection = None
        self._embedder = None
        self._ready = False

    # ── lazy init (avoids slow model load at import time) ─────────────────────

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(_EMBED_MODEL)
            self._client = chromadb.PersistentClient(path=str(_DB_PATH))
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._index_if_empty()
            self._ready = True
        except Exception as exc:  # pragma: no cover
            logger.error("RAG knowledge base init failed: %s", exc)
            self._ready = False

    def _index_if_empty(self) -> None:
        """Index all knowledge chunks when the collection is brand new."""
        if self._collection.count() > 0:
            logger.info("RAG: collection already has %d chunks — skipping re-index.", self._collection.count())
            return

        from llm_orchestrator.rag.knowledge_chunks import KNOWLEDGE_CHUNKS

        logger.info("RAG: Indexing %d knowledge chunks into ChromaDB …", len(KNOWLEDGE_CHUNKS))

        ids = [c["id"] for c in KNOWLEDGE_CHUNKS]
        texts = [f"{c['title']}\n\n{c['content']}" for c in KNOWLEDGE_CHUNKS]
        metadatas = [
            {
                "topic": c["topic"],
                "title": c["title"],
                "tags": ", ".join(c["tags"]),
            }
            for c in KNOWLEDGE_CHUNKS
        ]
        embeddings = self._embedder.encode(texts, show_progress_bar=False).tolist()

        # ChromaDB add in one batch
        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("RAG: Indexed %d chunks successfully.", len(KNOWLEDGE_CHUNKS))

    # ── public API ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = 4,
        topic_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search over the financial knowledge base.

        Returns a list of result dicts:
          - id: chunk id
          - title: chunk title
          - content: full text
          - topic: topic category
          - score: cosine similarity score (0–1, higher = more relevant)
        """
        self._ensure_ready()
        if not self._ready:
            return []

        query_embedding = self._embedder.encode([query], show_progress_bar=False).tolist()

        where_filter = {"topic": topic_filter} if topic_filter else None

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, self._collection.count() or n_results),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[dict[str, Any]] = []
        if not results["ids"] or not results["ids"][0]:
            return chunks

        for idx, chunk_id in enumerate(results["ids"][0]):
            doc = results["documents"][0][idx]
            meta = results["metadatas"][0][idx]
            distance = results["distances"][0][idx]
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity 0–1
            similarity = max(0.0, 1.0 - distance / 2)
            chunks.append(
                {
                    "id": chunk_id,
                    "title": meta.get("title", ""),
                    "content": doc,
                    "topic": meta.get("topic", ""),
                    "score": round(similarity, 3),
                }
            )

        return chunks

    @property
    def chunk_count(self) -> int:
        self._ensure_ready()
        if not self._ready or self._collection is None:
            return 0
        return self._collection.count()


# Singleton — one instance shared across the whole app lifetime
knowledge_base = FinancialKnowledgeBase()
