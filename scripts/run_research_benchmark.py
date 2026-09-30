"""Research benchmark: deterministic tools vs LLM arithmetic, same data for every arm.

Arms
  A_deployed : Stock-Mate chatbot as deployed (ResponseAgent, Groq). Gets the Upstox holdings, no price history.
  A_direct   : same Groq model, neutral prompt, temp 0, given ALL data the tool uses (holdings + price snapshot).
  B_tool     : Stock-Mate deterministic tool path, via the real router (0 LLM tokens).

Tiers
  1 = portfolio arithmetic from holdings (invested, current, P&L, P&L %, Buy/Hold/Trim/Exit)
  2 = single-stock time-series metrics on a pinned 3-month price snapshot
  3 = portfolio-level metrics asked implicitly ("my portfolio") - the in-app question

Ground truth comes from an independent pandas reference implementation (not the tool), so the tool is
verified too. Every raw response is saved to results.jsonl; external_prompt_pack.md holds the identical
prompts (holdings + prices pasted explicitly) for ChatGPT / Claude.

Run:    python scripts/run_research_benchmark.py --repeats 3
Resume: python scripts/run_research_benchmark.py --resume reports/benchmark_runs/<run>
        (reuses that run's pinned price snapshot; API quota errors stop the run instead of being scored)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402
from groq import Groq  # noqa: E402

from backend_api.core.config import settings  # noqa: E402
from llm_orchestrator.agents.response_agent import ResponseAgent  # noqa: E402
from llm_orchestrator.utils import quantitative_tools as qt  # noqa: E402

# Upstox holdings snapshot (the portfolio the in-app model sees).
UPSTOX_SNAPSHOT = [
    {"tradingsymbol": "IDEA", "exchange": "NSE", "quantity": 1, "average_price": 13, "last_price": 15},
    {"tradingsymbol": "YESBANK", "exchange": "NSE", "quantity": 1, "average_price": 23, "last_price": 23},
    {"tradingsymbol": "SUZLON", "exchange": "NSE", "quantity": 1, "average_price": 53, "last_price": 45},
]
# Realistic-size portfolio in the Upstox holdings schema (synthetic values).
REALISTIC = [
    {"tradingsymbol": "RELIANCE", "exchange": "NSE", "quantity": 42, "average_price": 2456.35, "last_price": 2891.70},
    {"tradingsymbol": "TCS", "exchange": "NSE", "quantity": 15, "average_price": 3620.10, "last_price": 3412.55},
    {"tradingsymbol": "HDFCBANK", "exchange": "NSE", "quantity": 60, "average_price": 1532.80, "last_price": 1678.25},
    {"tradingsymbol": "INFY", "exchange": "NSE", "quantity": 35, "average_price": 1478.90, "last_price": 1512.40},
    {"tradingsymbol": "ITC", "exchange": "NSE", "quantity": 250, "average_price": 412.65, "last_price": 438.90},
    {"tradingsymbol": "TATAMOTORS", "exchange": "NSE", "quantity": 80, "average_price": 958.40, "last_price": 712.15},
    {"tradingsymbol": "SBIN", "exchange": "NSE", "quantity": 120, "average_price": 598.25, "last_price": 811.60},
    {"tradingsymbol": "ZOMATO", "exchange": "NSE", "quantity": 400, "average_price": 142.30, "last_price": 238.75},
]
PORTFOLIOS = {"upstox_snapshot": UPSTOX_SNAPSHOT, "realistic_8": REALISTIC}

RULE = "Buy if total P&L% >= 10; Hold if 0 < P&L% < 10; Trim if -10 < P&L% <= 0; Exit if P&L% <= -10."
TIER1 = [  # (field, question, kind)
    ("total_invested", "What is the total invested value of my portfolio (sum of quantity x average price)?", "num"),
    ("total_current_value", "What is the total current value of my portfolio (sum of quantity x last price)?", "num"),
    ("total_pnl", "What is the total profit or loss of my portfolio in rupees?", "num"),
    ("total_pnl_pct", "What is the total percentage profit or loss of my portfolio?", "num"),
    ("decision", f"Classify the portfolio as Buy, Hold, Trim, or Exit using this rule: {RULE}", "label"),
]

TIER2_TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
FORMULAS = {
    "total_return": "Use (last close / first close - 1) x 100.",
    "max_drawdown": "Use the minimum over days of (value / running maximum value - 1) x 100, a negative number.",
    "var95": "Use daily simple returns; VaR = -(5th percentile of daily returns, linear interpolation) x 100.",
    "rsi14": "Use simple (non-smoothed) means of the last 14 daily close-to-close gains and losses: "
             "RSI = 100 - 100 / (1 + avgGain / avgLoss).",
    "sharpe": "Use daily simple returns; Sharpe = (mean daily return - 0.05/252) / sample standard deviation of "
              "daily returns x sqrt(252).",
}
TIER2 = [  # (metric, question template, tool result key, expected routed tool, exact tolerance)
    ("total_return", "What is the 3-month total return (%) of {t}?", "total_return_pct", "total_return", 0.02),
    ("max_drawdown", "What is the maximum drawdown (%) of {t} over the last 3 months?", "max_drawdown_pct", "drawdown", 0.02),
    ("var95", "What is the 1-day 95% historical Value at Risk (%) of {t} over the last 3 months?", "value_at_risk_pct", "var", 0.02),
    ("rsi14", "What is the 14-day RSI of {t} using the last 3 months of closes?", "rsi", "rsi", 0.1),
]
PORTFOLIO_DEF = "Portfolio value each day = sum of (quantity x that day's close) for my current holdings."
TIER3 = [  # (metric, question, tool result key, expected routed tool, exact tolerance)
    ("sharpe", "What is the Sharpe ratio of my portfolio over the last 3 months?", "sharpe_ratio", "sharpe", 0.005),
    ("var95", "What is the 1-day 95% historical Value at Risk (%) of my portfolio over the last 3 months?", "value_at_risk_pct", "var", 0.02),
    ("max_drawdown", "What is the maximum drawdown (%) of my portfolio over the last 3 months?", "max_drawdown_pct", "drawdown", 0.02),
    ("total_return", "What is the 3-month total return (%) of my portfolio?", "total_return_pct", "total_return", 0.02),
]

FINAL_NUM = "\nShow brief working, then end with one final line exactly in the form: FINAL: <number>"
FINAL_LABEL = "\nShow brief working, then end with one final line exactly in the form: FINAL: <Buy|Hold|Trim|Exit>"
LABELS = ("Buy", "Hold", "Trim", "Exit")
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
TPM = 7000  # stay under Groq free-tier 8k tokens/min


def normalize(rec: dict) -> dict:
    """Older rows stored numpy bools as the strings "True"/"False"; "False" would be truthy."""
    for key in ("strict", "lenient", "route_ok"):
        if isinstance(rec.get(key), str):
            rec[key] = rec[key] == "True"
    return rec


class QuotaExhausted(Exception):
    """The API refused every retry (e.g. Groq tokens-per-day). Never scored as a model failure."""


# ---------- independent reference implementation (ground truth) ----------
def ref_portfolio(holdings: list[dict]) -> dict:
    inv = sum(h["quantity"] * h["average_price"] for h in holdings)
    cur = sum(h["quantity"] * h["last_price"] for h in holdings)
    pct = (cur - inv) / inv * 100
    decision = "Buy" if pct >= 10 else "Hold" if pct > 0 else "Trim" if pct > -10 else "Exit"
    return {"total_invested": round(inv, 2), "total_current_value": round(cur, 2), "total_pnl": round(cur - inv, 2),
            "total_pnl_pct": round(pct, 2), "decision": decision}


def ref_metric(metric: str, s: pd.Series) -> float:
    r = s.pct_change().dropna()
    if metric == "total_return":
        return round((s.iloc[-1] / s.iloc[0] - 1) * 100, 2)
    if metric == "max_drawdown":
        return round(((s / s.cummax() - 1) * 100).min(), 2)
    if metric == "var95":
        return round(-np.percentile(r, 5) * 100, 2)
    if metric == "rsi14":
        d = s.diff().dropna()
        gain, loss = d.clip(lower=0).tail(14).mean(), (-d.clip(upper=0)).tail(14).mean()
        return round(100 - 100 / (1 + gain / loss), 2)
    if metric == "sharpe":
        return round((r.mean() - 0.05 / 252) / r.std() * math.sqrt(252), 4)
    raise ValueError(metric)


# ---------- parsing / scoring ----------
def _nums(text: str) -> list[float]:
    text = (text or "").replace("−", "-").replace("₹", "")
    return [float(m.replace(",", "")) for m in NUM_RE.findall(text)]


def parse_final(raw: str, kind: str):
    lines = re.findall(r"FINAL:\s*(.+)", raw or "")
    if not lines:
        return None
    if kind == "label":
        hit = [lab for lab in LABELS if lab.lower() in lines[-1].lower()]
        return hit[0] if len(hit) == 1 else None
    nums = _nums(lines[-1])
    return nums[0] if nums else None


def score(raw: str, kind: str, truth, tol: float) -> dict:
    parsed = parse_final(raw, kind)
    if kind == "label":
        return {"parsed": parsed, "strict": parsed == truth, "lenient": parsed == truth, "abs_err": None}
    strict = bool(parsed is not None and abs(parsed - truth) <= tol)  # plain bool: numpy bools serialise as text
    lenient = bool(any(abs(n - truth) <= tol for n in _nums(raw)))  # correct value stated anywhere in the reply
    abs_err = None if parsed is None else round(abs(parsed - truth), 4)
    return {"parsed": parsed, "strict": strict, "lenient": lenient, "abs_err": abs_err}


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


# ---------- LLM arms ----------
class Pacer:
    def __init__(self):
        self.last = 0.0
        self.wait = 0.0

    def before(self):
        time.sleep(max(0.0, self.last + self.wait - time.time()))

    def after(self, tokens: int):
        self.last, self.wait = time.time(), tokens / TPM * 60.0


pacer = Pacer()
agent = ResponseAgent()
client = Groq(api_key=settings.groq_api_key)


def call_deployed(question: str, holdings: list[dict]) -> dict:
    for attempt in range(8):
        pacer.before()
        t0 = time.perf_counter()
        reply = agent.reply(
            user_message=question,
            portfolio_context={"portfolio": {"holdings": holdings}, "rag_context": []},
            conversation_history=[],
            response_mode="quick",
        )
        latency = (time.perf_counter() - t0) * 1000.0
        if reply.token_usage:  # None => ResponseAgent fell back (rate limit / API error / empty reply)
            pacer.after(reply.token_usage.get("total_tokens") or 0)
            return {"raw": reply.raw_response or "", "latency_ms": round(latency, 2),
                    "tokens": reply.token_usage, "attempts": attempt + 1}
        print(f"    deployed fallback: {reply.answer[-160:]}", flush=True)
        if "tokens per day" in (reply.answer or "") or (attempt >= 2 and "RateLimitError" in (reply.answer or "")):
            raise QuotaExhausted(reply.answer[-200:])
        pacer.after(8000)
    raise QuotaExhausted("deployed chatbot fell back 8 times in a row")


def call_direct(system: str, question: str) -> dict:
    for attempt in range(8):
        pacer.before()
        t0 = time.perf_counter()
        try:
            r = client.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
                temperature=0,
                max_tokens=4096,
            )
        except Exception as exc:  # rate limit etc.
            print(f"    direct retry ({exc.__class__.__name__}: {str(exc)[:160]})", flush=True)
            if "tokens per day" in str(exc):
                raise QuotaExhausted(str(exc)[:300]) from exc
            pacer.after(8000)
            continue
        latency = (time.perf_counter() - t0) * 1000.0
        u = r.usage
        tokens = {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                  "total_tokens": u.total_tokens,
                  "reasoning_tokens": getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None)}
        pacer.after(u.total_tokens)
        return {"raw": r.choices[0].message.content or "", "finish_reason": r.choices[0].finish_reason,
                "latency_ms": round(latency, 2), "tokens": tokens, "attempts": attempt + 1}
    raise QuotaExhausted("direct call failed 8 times in a row")


def data_prompt(holdings: list[dict] | None = None, closes: pd.DataFrame | None = None) -> str:
    s = "You are a precise financial analyst. Use only the data provided and compute exactly."
    if holdings is not None:
        s += "\n\nMy portfolio holdings (JSON, Upstox schema):\n" + json.dumps(holdings)
    if closes is not None:
        s += "\n\nDaily closing prices, oldest first (CSV):\n" + closes.to_csv(float_format="%.2f", date_format="%Y-%m-%d")
    return s


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "reports" / "benchmark_runs"))
    ap.add_argument("--tiers", default="1,2,3")
    ap.add_argument("--resume", help="existing run directory to continue")
    args = ap.parse_args()

    rows: list[dict] = []
    if args.resume:
        out = Path(args.resume)
        meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
        for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines():
            rec = normalize(json.loads(line))
            if rec["arm"] != "B_tool" and (rec.get("failed") or rec.get("tokens") is None):
                with (out / "infra_failures.jsonl").open("a", encoding="utf-8") as f:
                    f.write(line + "\n")  # quota errors: logged for transparency, never scored
            else:
                rows.append(rec)
        (out / "results.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows), encoding="utf-8")
        meta.setdefault("resumed", []).append(datetime.now().isoformat())
    else:
        out = Path(args.out) / datetime.now().strftime("%Y-%m-%d_%H%M")
        out.mkdir(parents=True, exist_ok=True)
        meta = {"started": datetime.now().isoformat()}
    meta.update({"model": settings.groq_model, "provider": "groq", "repeats": args.repeats, "direct_temp": 0,
                 "direct_max_tokens": 4096, "risk_free_rate": 0.05})
    done = {(r["case"], r["arm"], r["repeat"]) for r in rows}
    log = (out / "results.jsonl").open("a", encoding="utf-8")

    def emit(rec: dict):
        rows.append(rec)
        log.write(json.dumps(rec, default=str) + "\n")
        log.flush()

    # Pinned 3-month price snapshot, rounded to 2dp, seeded into the tool cache: tool and LLMs see identical data.
    tier3_tickers = [qt._holding_ticker(h) for h in UPSTOX_SNAPSHOT]
    closes: dict[str, pd.Series] = {}
    if args.resume:  # identical data to the original run
        snap = pd.read_csv(out / "price_snapshot.csv", index_col=0, parse_dates=True)
        closes = {t: snap[t].dropna().round(2) for t in TIER2_TICKERS + tier3_tickers}
    else:
        fetch_ms = {}
        for t in TIER2_TICKERS + tier3_tickers:
            t0 = time.perf_counter()
            h = yf.download(t, period="3mo", interval="1d", progress=False, auto_adjust=False)
            fetch_ms[t] = round((time.perf_counter() - t0) * 1000.0, 2)
            closes[t] = h["Close"].squeeze().dropna().round(2)
        meta["cold_fetch_ms"] = fetch_ms
    qt._PRICE_HISTORY_CACHE.clear()
    for t, c in closes.items():
        qt._PRICE_HISTORY_CACHE[(t, "3mo", "1d")] = c.to_frame(name=t)
    if not args.resume:
        pd.DataFrame(closes).to_csv(out / "price_snapshot.csv")
    meta["snapshot_range"] = {t: [str(c.index[0].date()), str(c.index[-1].date()), len(c)] for t, c in closes.items()}

    cases = []
    if "1" in args.tiers:
        for pname, holdings in PORTFOLIOS.items():
            truth_all = ref_portfolio(holdings)
            for field, q, kind in TIER1:
                cases.append({"tier": 1, "case": f"{pname}:{field}", "question": q, "kind": kind, "tol": 0.011,
                              "truth": truth_all[field], "key": field, "expect": "portfolio_classification",
                              "holdings": holdings, "direct_system": data_prompt(holdings)})
    if "2" in args.tiers:
        for t in TIER2_TICKERS:
            for metric, qtpl, key, routed, tol in TIER2:
                cases.append({"tier": 2, "case": f"{t}:{metric}", "question": f"{qtpl.format(t=t)} {FORMULAS[metric]}",
                              "kind": "num", "tol": tol, "truth": ref_metric(metric, closes[t]), "key": key,
                              "expect": routed, "holdings": REALISTIC,
                              "direct_system": data_prompt(closes=closes[t].to_frame(name=t))})
    if "3" in args.tiers:
        frame = pd.concat([closes[t].rename(t) for t in tier3_tickers], axis=1, join="inner").dropna()
        value = sum(frame[qt._holding_ticker(h)] * h["quantity"] for h in UPSTOX_SNAPSHOT)
        for metric, q, key, routed, tol in TIER3:
            cases.append({"tier": 3, "case": f"portfolio:{metric}", "question": f"{q} {PORTFOLIO_DEF} {FORMULAS[metric]}",
                          "kind": "num", "tol": tol, "truth": ref_metric(metric, value), "key": key, "expect": routed,
                          "holdings": UPSTOX_SNAPSHOT, "direct_system": data_prompt(UPSTOX_SNAPSHOT, frame)})

    with (out / "external_prompt_pack.md").open("w", encoding="utf-8") as f:
        f.write("# External LLM prompt pack (ChatGPT / Claude)\n\nOne fresh chat per prompt, web browsing OFF. "
                "Record the FINAL line and wall-clock seconds.\n\n")
        for c in cases:
            suffix = FINAL_LABEL if c["kind"] == "label" else FINAL_NUM
            f.write(f"## {c['case']}\n\n```\n{c['direct_system']}\n\n{c['question']}{suffix}\n```\n\n")
    truth_file = out / "ground_truth.json"
    truth_now = {c["case"]: c["truth"] for c in cases}
    if args.resume and truth_file.exists():
        assert json.loads(truth_file.read_text()) == truth_now, "ground truth changed - snapshot mismatch"
    truth_file.write_text(json.dumps(truth_now, indent=2))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    for i, c in enumerate(cases, 1):
        q = c["question"] + (FINAL_LABEL if c["kind"] == "label" else FINAL_NUM)
        print(f"[{i}/{len(cases)}] {c['case']} truth={c['truth']}", flush=True)

        # B_tool: the real in-app path (router + tools), implicit question, holdings from the portfolio.
        if (c["case"], "B_tool", 0) not in done:
            run_tool(c, emit)
        try:
            for r in range(args.repeats):
                for arm, fn in (("A_deployed", lambda: call_deployed(q, c["holdings"])),
                                ("A_direct", lambda: call_direct(c["direct_system"], q))):
                    if (c["case"], arm, r) in done:
                        continue
                    resp = fn()
                    sc = score(resp["raw"], c["kind"], c["truth"], c["tol"])
                    emit({"tier": c["tier"], "case": c["case"], "arm": arm, "repeat": r, "truth": c["truth"], **sc, **resp})
                    print(f"    {arm} r{r}: parsed={sc['parsed']} strict={sc['strict']} "
                          f"lat={resp['latency_ms']} tok={(resp['tokens'] or {}).get('total_tokens')}", flush=True)
        except QuotaExhausted as exc:
            (out / "meta.json").write_text(json.dumps(meta, indent=2))
            print(f"\nSTOPPED (API quota): {exc}\nNothing was scored for the failed call. Resume later with:\n"
                  f"  python scripts/run_research_benchmark.py --resume \"{out}\"", flush=True)
            return

    meta["finished"] = datetime.now().isoformat()
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "summary.md").write_text(summarize(rows, meta), encoding="utf-8")
    print(f"\nDone -> {out}")


def run_tool(c: dict, emit) -> None:
    t0 = time.perf_counter()
    res = qt.execute_quantitative_tools(c["question"], holdings=c["holdings"])
    tool_ms = (time.perf_counter() - t0) * 1000.0
    ok = bool(res) and res[0]["status"] == "ok"
    result = res[0]["result"] if ok else {}
    val = result.get(c["key"])
    route_ok = ok and res[0]["tool"] == c["expect"] and (c["tier"] != 3 or result.get("scope") == "portfolio")
    strict = bool(val is not None and (val == c["truth"] if c["kind"] == "label" else abs(val - c["truth"]) <= c["tol"]))
    emit({"tier": c["tier"], "case": c["case"], "arm": "B_tool", "repeat": 0, "truth": c["truth"], "parsed": val,
          "strict": strict, "lenient": strict,
          "abs_err": None if val is None or c["kind"] == "label" else round(abs(val - c["truth"]), 4),
          "latency_ms": round(tool_ms, 4), "tokens": {"total_tokens": 0},
          "routed": f"{res[0]['tool']}:{res[0]['status']}" if res else None, "route_ok": route_ok,
          "tool_error": None if ok else (res[0].get("error") if res else "no tool selected")})
    print(f"    B_tool: {val} strict={strict} routed={res[0]['tool'] if res else None}", flush=True)


TIER_TITLES = {1: "Tier 1 - portfolio arithmetic (holdings only)",
               2: "Tier 2 - single-stock time-series metrics (pinned 3-month prices)",
               3: "Tier 3 - portfolio-level metrics, implicit 'my portfolio' question"}


def summarize(rows: list[dict], meta: dict) -> str:
    L = [f"# Benchmark summary ({meta.get('started', '?')[:16]})", "",
         f"Model: `{meta['model']}` via Groq. Repeats per LLM case: {meta['repeats']}. "
         "Ground truth: independent pandas reference implementation.", ""]
    for tier in (1, 2, 3):
        tr = [r for r in rows if r["tier"] == tier]
        if not tr:
            continue
        arms = list(dict.fromkeys(r["arm"] for r in tr))
        L += [f"## {TIER_TITLES[tier]}", "",
              "| Arm | n | Exact acc % (95% CI) | Correct value anywhere % | No final answer % | Median abs err | "
              "Latency median / mean / p95 ms | Mean tokens |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for arm in arms:
            ar = [r for r in tr if r["arm"] == arm]
            n = len(ar)
            k = sum(bool(r["strict"]) for r in ar)
            lo, hi = wilson(k, n)
            errs = [r["abs_err"] for r in ar if r.get("abs_err") is not None]
            lats = sorted(r["latency_ms"] for r in ar if r.get("latency_ms") is not None)
            toks = [(r.get("tokens") or {}).get("total_tokens") or 0 for r in ar]
            p95 = lats[min(len(lats) - 1, int(0.95 * len(lats)))] if lats else None
            lat_s = f"{st.median(lats):.2f} / {st.mean(lats):.2f} / {p95:.2f}" if lats else "n/a"
            L.append(f"| {arm} | {n} | {100 * k / n:.1f} ({lo}-{hi}) | {100 * sum(bool(r['lenient']) for r in ar) / n:.1f} | "
                     f"{100 * sum(r['parsed'] is None for r in ar) / n:.1f} | {st.median(errs) if errs else 'n/a'} | "
                     f"{lat_s} | {st.mean(toks):.0f} |")
        cons = []
        for arm in arms:
            if arm == "B_tool":
                continue
            per_case: dict[str, set] = {}
            for r in tr:
                if r["arm"] == arm:
                    per_case.setdefault(r["case"], set()).add(r["parsed"])
            cons.append(f"{arm} {sum(len(v) == 1 for v in per_case.values())}/{len(per_case)}")
        routes = [r for r in tr if r["arm"] == "B_tool"]
        L += ["", "Same answer on every repeat (cases): " + ", ".join(cons),
              f"Tool routed to the correct tool/scope: {sum(bool(r['route_ok']) for r in routes)}/{len(routes)}", "",
              "| Case | Truth | " + " | ".join(arms) + " |", "|---|---:|" + "---|" * len(arms)]
        for case in dict.fromkeys(r["case"] for r in tr):
            cells = [", ".join("-" if r["parsed"] is None else str(r["parsed"])
                               for r in tr if r["case"] == case and r["arm"] == arm) for arm in arms]
            L.append(f"| {case} | {next(r['truth'] for r in tr if r['case'] == case)} | " + " | ".join(cells) + " |")
        L.append("")
    if meta.get("cold_fetch_ms"):
        L.append("Cold market-data fetch per ticker (yfinance, ms): " + json.dumps(meta["cold_fetch_ms"]))
    L += ["Price snapshot [first, last, n]: " + json.dumps(meta.get("snapshot_range")), ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
