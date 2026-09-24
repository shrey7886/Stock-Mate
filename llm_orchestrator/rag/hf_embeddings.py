from __future__ import annotations

import logging

import requests

from backend_api.core.config import settings

logger = logging.getLogger(__name__)

HF_MODEL_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Embed texts via Hugging Face's hosted Inference API (same model as before,
    all-MiniLM-L6-v2) instead of loading sentence-transformers/torch locally —
    keeps the backend's memory footprint small enough for free-tier hosting.

    Returns a list of embedding vectors (same order as `texts`), or None on failure.
    """
    if not texts:
        return []
    if not settings.hf_api_token:
        logger.warning("HF_API_TOKEN not set — RAG embeddings unavailable.")
        return None

    try:
        resp = requests.post(
            HF_MODEL_URL,
            headers={"Authorization": f"Bearer {settings.hf_api_token}"},
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("HF embedding call failed: %s", exc)
        return None
