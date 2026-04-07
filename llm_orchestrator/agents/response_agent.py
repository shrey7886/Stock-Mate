"""
ResponseAgent: calls Groq (default/free) or OpenAI based on LLM_PROVIDER config.
Groq uses the same OpenAI-compatible SDK interface.
Returns a structured ChatReply.
"""
from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

from groq import Groq
from openai import OpenAI

from backend_api.core.config import settings

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "system_prompt.txt"
_SYSTEM_PROMPT_TEMPLATE: str = _PROMPT_PATH.read_text(encoding="utf-8")


@dataclass
class ChatReply:
    answer: str
    action_tag: str        # Hold | Trim | Add | Watch | Rebalance | None
    why: str
    risk_note: str
    confidence: str        # low | medium | high
    detected_intent: str = "unknown"
    confidence_score: float = 0.0
    next_steps: list[str] | None = None
    proactive_insights: list[dict] | None = None
    portfolio_health: dict | None = None
    raw_response: str = ""


_EXTRACTION_INSTRUCTION = textwrap.dedent("""

    ---
    After your conversational reply, output a JSON block on its own line (nothing else on that line):
    {"action_tag": "...", "why": "...", "risk_note": "...", "confidence": "..."}
    Use action_tag values: Hold | Trim | Add | Watch | Rebalance | None
    Use confidence values: low | medium | high
""")


def _build_client() -> tuple[object, str]:
    """
    Returns (client, model) based on LLM_PROVIDER setting.
    Priority: groq → openai → None (fallback mode).
    """
    provider = (settings.llm_provider or "groq").lower()

    if provider == "groq" and settings.groq_api_key:
        return Groq(api_key=settings.groq_api_key), settings.groq_model

    if provider == "openai" and settings.openai_api_key:
        return OpenAI(api_key=settings.openai_api_key), settings.openai_model

    # auto-detect: try groq first, then openai
    if settings.groq_api_key:
        return Groq(api_key=settings.groq_api_key), settings.groq_model
    if settings.openai_api_key:
        return OpenAI(api_key=settings.openai_api_key), settings.openai_model

    return None, ""


class ResponseAgent:
    def __init__(self) -> None:
        self._client, self._model = _build_client()

    def is_available(self) -> bool:
        return self._client is not None

    def reply(
        self,
        *,
        user_message: str,
        portfolio_context: dict,
        conversation_history: list[dict],
    ) -> ChatReply:
        if self._client is None:
            return self._fallback_reply(
                portfolio_context=portfolio_context,
                reason="No LLM API key configured. Add GROQ_API_KEY to .env (free at console.groq.com).",
            )

        # Extract RAG chunks before serializing full context
        rag_chunks: list[str] = portfolio_context.pop("rag_context", [])
        rag_str = (
            "\n\n---\n\n".join(rag_chunks)
            if rag_chunks
            else "(No additional knowledge retrieved for this query.)"
        )

        context_str = json.dumps(portfolio_context, ensure_ascii=False, indent=2)
        history_str = self._format_history(conversation_history)
        system_msg = (
            _SYSTEM_PROMPT_TEMPLATE
            .replace("{rag_context}", rag_str)
            .replace("{portfolio_context}", context_str)
            .replace("{conversation_history}", history_str or "(no prior conversation)")
        )

        messages = [{"role": "system", "content": system_msg + _EXTRACTION_INSTRUCTION}]
        for turn in conversation_history:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.5,
                max_tokens=800,
            )
            raw = response.choices[0].message.content or ""
            return self._parse(raw)
        except Exception as exc:
            return self._fallback_reply(
                portfolio_context=portfolio_context,
                reason=f"LLM temporarily unavailable ({exc.__class__.__name__}: {exc}).",
            )

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        lines = []
        for turn in history:
            role = turn.get("role", "user").capitalize()
            lines.append(f"{role}: {turn.get('content', '')}")
        return "\n".join(lines)

    @staticmethod
    def _parse(raw: str) -> ChatReply:
        """Split conversational text from the trailing JSON metadata block."""
        answer_lines = []
        meta: dict = {}

        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    meta = json.loads(stripped)
                except json.JSONDecodeError:
                    answer_lines.append(line)
            else:
                answer_lines.append(line)

        return ChatReply(
            answer="\n".join(answer_lines).strip(),
            action_tag=meta.get("action_tag", "None"),
            why=meta.get("why", ""),
            risk_note=meta.get("risk_note", ""),
            confidence=meta.get("confidence", "medium"),
            raw_response=raw,
        )

    @staticmethod
    def _fallback_reply(*, portfolio_context: dict, reason: str) -> ChatReply:
        portfolio = portfolio_context.get("portfolio", {})
        total_pnl_pct = float(portfolio.get("total_pnl_pct") or 0.0)
        invested = float(portfolio.get("total_invested") or 0.0)
        current = float(portfolio.get("total_current_value") or 0.0)
        holdings_count = int(portfolio.get("holdings_count") or 0)

        direction = "up" if total_pnl_pct >= 0 else "down"
        answer = (
            f"I can still give you a quick portfolio snapshot while AI chat is offline. "
            f"You currently have {holdings_count} holdings with about ₹{current:,.2f} current value "
            f"on ₹{invested:,.2f} invested — {direction} {abs(total_pnl_pct):.2f}% overall. "
            f"({reason})"
        )

        return ChatReply(
            answer=answer,
            action_tag="Watch" if total_pnl_pct < 0 else "Hold",
            why="Fallback summary generated from live portfolio data.",
            risk_note="This is insight, not financial advice.",
            confidence="low",
            raw_response=answer,
        )


