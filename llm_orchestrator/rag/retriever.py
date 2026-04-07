"""
High-level RAG retriever used by the pipeline to fetch relevant knowledge chunks
for any user query + intent combination.
"""
from __future__ import annotations

import logging

from llm_orchestrator.rag.knowledge_base import knowledge_base

logger = logging.getLogger(__name__)

# Intent → preferred topic filter (helps narrow results when relevant)
_INTENT_TOPIC_MAP: dict[str, str | None] = {
    "tax_analysis": "tax",
    "sip_advice": "sip",
    "goal_tracking": "sip",
    "portfolio_health": "portfolio",
    "rebalancing": "portfolio",
    "allocation": "portfolio",
    "risk": "risk",
    "market_overview": "macro",
    "stock_comparison": "fundamental",
    "performance": None,
    "portfolio_summary": None,
    "position_analysis": None,
    "watchlist": None,
    "next_action": None,
}


class RAGRetriever:
    """
    Retrieves relevant financial knowledge chunks given a user query + intent.
    Returns formatted text blocks ready to be injected into the LLM context.
    """

    def retrieve(
        self,
        query: str,
        intent: str = "",
        n_results: int = 3,
        min_score: float = 0.25,
    ) -> list[str]:
        """
        Args:
            query:      Raw user message / rephrased query.
            intent:     Classified intent string (from intent_router).
            n_results:  Max number of chunks to return.
            min_score:  Minimum cosine similarity to include a chunk (0–1).

        Returns:
            List of formatted strings, each being a self-contained knowledge excerpt.
        """
        try:
            topic_filter = _INTENT_TOPIC_MAP.get(intent)
            results = knowledge_base.search(query, n_results=n_results + 2, topic_filter=topic_filter)

            # Fallback: if topic-filtered search yields too few results, search without filter
            if topic_filter and len([r for r in results if r["score"] >= min_score]) < 2:
                results = knowledge_base.search(query, n_results=n_results + 2)

            filtered = [r for r in results if r["score"] >= min_score][:n_results]

            if not filtered:
                logger.debug("RAG: no relevant chunks found for query=%r (intent=%s)", query[:60], intent)
                return []

            formatted: list[str] = []
            for chunk in filtered:
                formatted.append(
                    f"[Financial Knowledge — {chunk['title']}]\n{chunk['content']}"
                )

            logger.debug(
                "RAG: retrieved %d chunks for intent=%s (scores: %s)",
                len(formatted),
                intent,
                [r["score"] for r in filtered],
            )
            return formatted

        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)
            return []


# Singleton
rag_retriever = RAGRetriever()
