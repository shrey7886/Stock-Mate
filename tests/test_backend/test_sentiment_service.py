"""Run directly: python tests/test_backend/test_sentiment_service.py"""
from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

from backend_api.services import sentiment_service


def test_no_token_returns_all_none():
    with patch.object(sentiment_service, "settings", replace(sentiment_service.settings, hf_api_token="")):
        assert sentiment_service.analyze_sentiment_batch(["a", "b"]) == [None, None]


def test_empty_input_returns_empty():
    assert sentiment_service.analyze_sentiment_batch([]) == []


def test_happy_path_picks_highest_score_label():
    fake_response = [
        [{"label": "positive", "score": 0.9}, {"label": "neutral", "score": 0.08}, {"label": "negative", "score": 0.02}],
        [{"label": "negative", "score": 0.7}, {"label": "neutral", "score": 0.2}, {"label": "positive", "score": 0.1}],
    ]

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return fake_response

    fake_settings = replace(sentiment_service.settings, hf_api_token="fake-token")
    with patch.object(sentiment_service, "settings", fake_settings), \
         patch.object(sentiment_service.requests, "post", return_value=FakeResp()):
        result = sentiment_service.analyze_sentiment_batch(["good news", "bad news"])

    assert result == [{"label": "positive", "score": 0.9}, {"label": "negative", "score": 0.7}]


def test_api_failure_returns_all_none():
    fake_settings = replace(sentiment_service.settings, hf_api_token="fake-token")
    with patch.object(sentiment_service, "settings", fake_settings), \
         patch.object(sentiment_service.requests, "post", side_effect=Exception("timeout")):
        assert sentiment_service.analyze_sentiment_batch(["x"]) == [None]


if __name__ == "__main__":
    test_no_token_returns_all_none()
    test_empty_input_returns_empty()
    test_happy_path_picks_highest_score_label()
    test_api_failure_returns_all_none()
    print("ok")
