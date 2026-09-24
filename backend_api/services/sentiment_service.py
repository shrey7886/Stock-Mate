from __future__ import annotations

import logging

import requests

from backend_api.core.config import settings

logger = logging.getLogger(__name__)

HF_MODEL_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"


def analyze_sentiment_batch(texts: list[str]) -> list[dict | None]:
    """Score each text's sentiment via the hosted FinBERT model on Hugging Face's Inference API.

    Returns a list the same length as `texts`; each entry is either
    {"label": "positive"|"negative"|"neutral", "score": float} or None if scoring
    was unavailable for that text (no token configured, API error, timeout, etc).
    News digest must keep working even when sentiment scoring fails.
    """
    if not texts:
        return []
    if not settings.hf_api_token:
        return [None] * len(texts)

    try:
        resp = requests.post(
            HF_MODEL_URL,
            headers={"Authorization": f"Bearer {settings.hf_api_token}"},
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=20,
        )
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:
        logger.warning("HF sentiment scoring failed: %s", exc)
        return [None] * len(texts)

    if not isinstance(raw, list) or len(raw) != len(texts):
        return [None] * len(texts)

    results: list[dict | None] = []
    for per_text_scores in raw:
        if not isinstance(per_text_scores, list) or not per_text_scores:
            results.append(None)
            continue
        best = max(per_text_scores, key=lambda item: item.get("score", 0.0))
        results.append({"label": best.get("label"), "score": round(float(best.get("score", 0.0)), 4)})
    return results
