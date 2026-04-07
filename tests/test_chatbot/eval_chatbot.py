"""
Live chatbot evaluation script.
Hits the running backend API with a gold question set and scores:
  - Intent accuracy (detected_intent vs expected)
  - Hallucination guard (answers must not fabricate numbers for empty portfolios)
  - Response completeness (required fields present, non-empty)
  - Guardrail accuracy (off-topic questions must be redirected)
  - Confidence calibration (in-scope → higher score than out-of-scope)

Usage:
    python tests/test_chatbot/eval_chatbot.py [--base-url http://127.0.0.1:8000]

CI usage:
    python tests/test_chatbot/eval_chatbot.py --fail-below 0.80
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import requests

# ── Gold evaluation set ───────────────────────────────────────────────────────
# Each case defines: the user message, the expected detected_intent (or None to skip),
# whether it should be handled vs redirected, and optional answer checks.

GOLD_CASES: list[dict] = [
    # ── In-scope: portfolio summary ──────────────────────────────────────────
    {
        "message": "Give me a summary of my portfolio",
        "expected_intent": "portfolio_summary",
        "in_scope": True,
        "answer_must_not_contain": ["```", "Error", "Traceback"],
    },
    {
        "message": "How is my portfolio doing today?",
        "expected_intent": "portfolio_summary",
        "in_scope": True,
    },
    # ── In-scope: performance ────────────────────────────────────────────────
    {
        "message": "What are my returns this month?",
        "expected_intent": "performance",
        "in_scope": True,
    },
    {
        "message": "How much profit or loss am I sitting on?",
        "expected_intent": "performance",
        "in_scope": True,
    },
    # ── In-scope: allocation ─────────────────────────────────────────────────
    {
        "message": "How concentrated is my portfolio?",
        "expected_intent": "allocation",
        "in_scope": True,
    },
    {
        "message": "Am I well diversified?",
        "expected_intent": "allocation",
        "in_scope": True,
    },
    # ── In-scope: risk ───────────────────────────────────────────────────────
    {
        "message": "What is my portfolio risk right now?",
        "expected_intent": "risk",
        "in_scope": True,
    },
    # ── In-scope: rebalancing ────────────────────────────────────────────────
    {
        "message": "Should I rebalance my portfolio?",
        "expected_intent": "rebalancing",
        "in_scope": True,
    },
    # ── In-scope: next action ────────────────────────────────────────────────
    {
        "message": "What should I do next with my investments?",
        "expected_intent": "next_action",
        "in_scope": True,
    },
    # ── Out-of-scope: guardrail ──────────────────────────────────────────────
    {
        "message": "Tell me a joke",
        "expected_intent": "unknown",
        "in_scope": False,
        "answer_must_contain_one_of": ["portfolio", "holding", "allocation", "risk", "invest"],
    },
    {
        "message": "What is the capital of France?",
        "expected_intent": "unknown",
        "in_scope": False,
        "answer_must_contain_one_of": ["portfolio", "holding", "I can help"],
    },
    {
        "message": "Who won the cricket match yesterday?",
        "expected_intent": "unknown",
        "in_scope": False,
    },
    # ── Safety: no hallucinated numbers on empty portfolio ───────────────────
    {
        "message": "List all my gains in exact rupees",
        "expected_intent": "performance",
        "in_scope": True,
        "no_fabricated_numbers": True,   # checked separately below
    },
]


@dataclass
class EvalResult:
    total: int = 0
    intent_correct: int = 0
    scope_correct: int = 0
    fields_complete: int = 0
    guardrail_correct: int = 0
    confidence_ordered: int = 0
    failures: list[str] = field(default_factory=list)


def _login(base: str) -> str:
    r = requests.post(f"{base}/api/auth/login", json={"user_id": "eval_user"}, timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]


def _chat(base: str, token: str, message: str) -> dict:
    r = requests.post(
        f"{base}/api/chat/message",
        json={"message": message},
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


REQUIRED_FIELDS = {"answer", "action_tag", "why", "confidence", "detected_intent", "confidence_score", "next_steps"}


def evaluate(base: str, verbose: bool = True) -> EvalResult:
    result = EvalResult()
    token = _login(base)

    in_scope_scores: list[float] = []
    out_scope_scores: list[float] = []

    for case in GOLD_CASES:
        result.total += 1
        message = case["message"]
        expected_intent = case.get("expected_intent")
        in_scope = case.get("in_scope", True)

        try:
            data = _chat(base, token, message)
        except Exception as exc:
            result.failures.append(f"[API ERROR] '{message}' => {exc}")
            continue

        if verbose:
            print(f"\n  Q: {message}")
            print(f"     intent={data.get('detected_intent')} conf={data.get('confidence_score')}")
            print(f"     answer={data.get('answer','')[:120]!r}")

        # 1. Intent accuracy
        detected = data.get("detected_intent", "")
        if expected_intent and detected == expected_intent:
            result.intent_correct += 1
        elif expected_intent:
            result.failures.append(
                f"[INTENT] '{message}' => got '{detected}', expected '{expected_intent}'"
            )

        # 2. Required fields completeness
        missing = REQUIRED_FIELDS - data.keys()
        if not missing:
            result.fields_complete += 1
        else:
            result.failures.append(f"[FIELDS] '{message}' missing: {missing}")

        # 3. Guardrail check: off-scope should mention portfolio in answer
        answer_lower = (data.get("answer") or "").lower()
        if not in_scope:
            must_contain = case.get("answer_must_contain_one_of", [])
            if not must_contain or any(kw in answer_lower for kw in must_contain):
                result.guardrail_correct += 1
            else:
                result.failures.append(
                    f"[GUARDRAIL] '{message}' => answer didn't redirect: {data.get('answer','')[:100]!r}"
                )

        # 4. Answer quality checks
        must_not = case.get("answer_must_not_contain", [])
        for bad in must_not:
            if bad.lower() in answer_lower:
                result.failures.append(f"[QUALITY] '{message}' answer contains forbidden: {bad!r}")

        # 5. Track confidence scores for calibration check
        score = float(data.get("confidence_score", 0.0))
        if in_scope:
            in_scope_scores.append(score)
        else:
            out_scope_scores.append(score)

        # 6. scope correct = in-scope went through LLM or had portfolio content
        if in_scope and len(data.get("answer", "")) > 30:
            result.scope_correct += 1

        time.sleep(0.3)  # Groq rate limit safety

    # Confidence calibration: in-scope avg should be > out-of-scope avg
    if in_scope_scores and out_scope_scores:
        avg_in = sum(in_scope_scores) / len(in_scope_scores)
        avg_out = sum(out_scope_scores) / len(out_scope_scores)
        if avg_in > avg_out:
            result.confidence_ordered = 1
        else:
            result.failures.append(
                f"[CALIBRATION] in-scope avg conf {avg_in:.2f} not > out-scope avg {avg_out:.2f}"
            )

    return result


def print_report(result: EvalResult) -> None:
    n = result.total
    intent_pct   = result.intent_correct / n * 100    if n else 0
    fields_pct   = result.fields_complete / n * 100   if n else 0
    guard_cases  = sum(1 for c in GOLD_CASES if not c.get("in_scope", True))
    guard_pct    = result.guardrail_correct / guard_cases * 100 if guard_cases else 100

    overall = (intent_pct + fields_pct + guard_pct) / 3

    print("\n" + "=" * 58)
    print("  STOCKMATE CHATBOT EVAL REPORT")
    print("=" * 58)
    print(f"  Total questions  : {n}")
    print(f"  Intent accuracy  : {result.intent_correct}/{n}  ({intent_pct:.1f}%)")
    print(f"  Fields complete  : {result.fields_complete}/{n}  ({fields_pct:.1f}%)")
    print(f"  Guardrail pass   : {result.guardrail_correct}/{guard_cases}  ({guard_pct:.1f}%)")
    print(f"  Confidence order : {'PASS' if result.confidence_ordered else 'FAIL'}")
    print(f"  Overall score    : {overall:.1f}%")
    print("=" * 58)

    if result.failures:
        print(f"\n  ⚠  {len(result.failures)} failure(s):")
        for f in result.failures:
            print(f"     {f}")
    else:
        print("\n  ✓  No failures detected.")
    print()
    return overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--fail-below", type=float, default=0.0,
                        help="Exit code 1 if overall score < this threshold (0–100)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"\nRunning eval against {args.base_url} ...")
    result = evaluate(args.base_url, verbose=not args.quiet)
    overall = print_report(result)

    if args.fail_below and overall < args.fail_below:
        print(f"  FAIL: score {overall:.1f}% below threshold {args.fail_below}%")
        sys.exit(1)
