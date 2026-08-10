"""Run directly: python tests/test_backend/test_hf_embeddings.py"""
from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

from llm_orchestrator.rag import hf_embeddings


def test_empty_input_returns_empty():
    assert hf_embeddings.embed_texts([]) == []


def test_no_token_returns_none():
    with patch.object(hf_embeddings, "settings", replace(hf_embeddings.settings, hf_api_token="")):
        assert hf_embeddings.embed_texts(["a"]) is None


def test_happy_path_returns_vectors():
    fake_vectors = [[0.1, 0.2], [0.3, 0.4]]

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return fake_vectors

    fake_settings = replace(hf_embeddings.settings, hf_api_token="fake-token")
    with patch.object(hf_embeddings, "settings", fake_settings), \
         patch.object(hf_embeddings.requests, "post", return_value=FakeResp()):
        assert hf_embeddings.embed_texts(["x", "y"]) == fake_vectors


def test_api_failure_returns_none():
    fake_settings = replace(hf_embeddings.settings, hf_api_token="fake-token")
    with patch.object(hf_embeddings, "settings", fake_settings), \
         patch.object(hf_embeddings.requests, "post", side_effect=Exception("timeout")):
        assert hf_embeddings.embed_texts(["x"]) is None


if __name__ == "__main__":
    test_empty_input_returns_empty()
    test_no_token_returns_none()
    test_happy_path_returns_vectors()
    test_api_failure_returns_none()
    print("ok")
