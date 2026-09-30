"""Generate paper/generated_results.tex from a benchmark run, so no number in the paper is hand-copied.

python scripts/make_paper_tables.py reports/benchmark_runs/<run> [--external <scored csv> ...]

Defines \\res{<key>} values (e.g. \\res{T1-A_direct-acc}) and ready-made tables. Keys that are missing
render as a bold "??" in the PDF, so an incomplete run is visible, never silently wrong.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARM_LABEL = {"B_tool": "Foleo tools (ours)", "A_deployed": "Groq chatbot (no prices)", "A_direct": "Groq + full data"}
TIER_LABEL = {1: "Portfolio arithmetic", 2: "Stock time-series metrics", 3: "Portfolio risk metrics"}
TOLERANCE = {"rsi14": 0.1, "sharpe": 0.005}
EXTERNAL_LABEL = {"chatgpt-gpt5.6-luna": "ChatGPT (GPT-5.6 Luna)", "claude-sonnet5-low": "Claude Sonnet 5 (low)"}


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def tex(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def fmt(x: float | None, nd: int = 1) -> str:
    return "--" if x is None else f"{x:,.{nd}f}"


def arm_stats(rows: list[dict]) -> dict:
    n = len(rows)
    k = sum(bool(r["strict"]) for r in rows)
    lo, hi = wilson(k, n)
    lats = sorted(r["latency_ms"] for r in rows if r.get("latency_ms") is not None)
    toks = [(r.get("tokens") or {}).get("total_tokens") or 0 for r in rows]
    per_case: dict[str, set] = {}
    for r in rows:
        per_case.setdefault(r["case"], set()).add(r["parsed"])
    return {
        "n": n, "k": k, "acc": 100 * k / n, "lo": lo, "hi": hi,
        "noans": 100 * sum(r["parsed"] is None for r in rows) / n,
        "medlat": st.median(lats) if lats else None, "meanlat": st.mean(lats) if lats else None,
        "p95lat": lats[min(len(lats) - 1, int(0.95 * len(lats)))] if lats else None,
        "tok": st.mean(toks), "medtok": st.median(toks), "maxtok": max(toks),
        "cons": sum(len(v) == 1 for v in per_case.values()), "cases": len(per_case),
        "trunc": sum(r.get("finish_reason") == "length" for r in rows),
        "miss": n - k,
        "wrong": sum(r["parsed"] is not None and not r["strict"] for r in rows),
        "nofinal": sum(r["parsed"] is None for r in rows),
    }


def score_external(path: Path, truth: dict) -> list[dict]:
    out = []
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if not r["final_answer"].strip():
            continue
        case, expected = r["case"], truth[r["case"]]
        metric = case.split(":")[1]
        tier = 1 if case.startswith(("upstox_snapshot", "realistic_8")) else 3 if case.startswith("portfolio") else 2
        ans = r["final_answer"].replace("−", "-").replace("₹", "").replace(",", "")
        if isinstance(expected, str):
            hits = [lab for lab in ("Buy", "Hold", "Trim", "Exit") if lab.lower() in ans.lower()]
            got = hits[0] if len(hits) == 1 else None
            ok = got == expected
        else:
            m = re.search(r"-?\d+(?:\.\d+)?", ans)
            got = float(m.group()) if m else None
            ok = got is not None and abs(got - expected) <= TOLERANCE.get(metric, 0.011 if tier == 1 else 0.02)
        out.append({"model": r["model"], "tier": tier, "case": case, "ok": ok, "got": got, "truth": expected})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--external", nargs="*", default=[])
    ap.add_argument("--out", default=str(ROOT / "paper" / "generated_results.tex"))
    args = ap.parse_args()
    run = Path(args.run)
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in rows:  # older rows stored numpy bools as "True"/"False" text; "False" would be truthy
        for key in ("strict", "lenient", "route_ok"):
            if isinstance(r.get(key), str):
                r[key] = r[key] == "True"
    truth = json.loads((run / "ground_truth.json").read_text())
    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
    infra = (run / "infra_failures.jsonl")
    n_infra = len(infra.read_text().splitlines()) if infra.exists() else 0

    vals: dict[str, str] = {}
    expected_cases = {1: 10, 2: 12, 3: 4}
    complete = True
    for tier in (1, 2, 3):
        tr = [r for r in rows if r["tier"] == tier]
        have = len({r["case"] for r in tr if r["arm"] != "B_tool"})
        complete &= have == expected_cases[tier] and all(
            sum(1 for r in tr if r["arm"] == a and r["case"] == c) == meta.get("repeats", 3)
            for a in ("A_deployed", "A_direct") for c in {r["case"] for r in tr})
        for arm in ("B_tool", "A_deployed", "A_direct"):
            ar = [r for r in tr if r["arm"] == arm]
            if not ar:
                continue
            s = arm_stats(ar)
            key = f"T{tier}-{arm}"
            for name, nd in (("acc", 1), ("lo", 1), ("hi", 1), ("noans", 1), ("medlat", 1), ("meanlat", 1),
                             ("p95lat", 1), ("tok", 0), ("medtok", 0)):
                vals[f"{key}-{name}"] = fmt(s[name], 3 if arm == "B_tool" and "lat" in name else nd)
            for name in ("n", "k", "cons", "cases", "maxtok", "trunc", "miss", "wrong", "nofinal"):
                vals[f"{key}-{name}"] = f"{s[name]:,}"
        tool = [r for r in tr if r["arm"] == "B_tool"]
        vals[f"T{tier}-route"] = f"{sum(bool(r['route_ok']) for r in tool)}/{len(tool)}"
    for arm in ("B_tool", "A_deployed", "A_direct"):
        ar = [r for r in rows if r["arm"] == arm]
        if ar:
            vals[f"ALL-{arm}-k"], vals[f"ALL-{arm}-n"] = str(sum(bool(r["strict"]) for r in ar)), str(len(ar))
    nodata = [r for r in rows if r["arm"] == "A_deployed" and r["tier"] in (2, 3)]  # chatbot had no price history
    if nodata:
        vals["NODATA-n"] = str(len(nodata))
        vals["NODATA-nofinal"] = str(sum(r["parsed"] is None for r in nodata))
        vals["NODATA-valued"] = str(sum(r["parsed"] is not None for r in nodata))
    vals["infra"] = str(n_infra)
    fetch = run / "fetch_latency.json"  # measured separately: market-data retrieval is outside the tool timing
    if fetch.exists():
        f = json.loads(fetch.read_text())
        vals["fetch-med"], vals["fetch-p95"] = fmt(f["median_ms"], 0), fmt(f["p95_ms"], 0)
    vals["model"] = tex(meta.get("model", "??"))
    vals["repeats"] = str(meta.get("repeats", "??"))

    ext_rows: list[dict] = []
    for path in args.external:
        ext_rows += score_external(Path(path), truth)
    for model in dict.fromkeys(r["model"] for r in ext_rows):
        for tier in (1, 2, 3, 0):
            er = [r for r in ext_rows if r["model"] == model and (tier == 0 or r["tier"] == tier)]
            if er:
                vals[f"X-{model}-{tier or 'all'}"] = f"{sum(r['ok'] for r in er)}/{len(er)}"

    L = ["% AUTO-GENERATED by scripts/make_paper_tables.py - do not edit by hand.",
         f"% Source run: {run.name}; complete={complete}",
         r"\makeatletter"]
    for k, v in vals.items():
        L.append(rf"\expandafter\def\csname res@{k}\endcsname{{{v}}}")
    L += [r"\makeatother", rf"\def\RunComplete{{{'1' if complete else '0'}}}", ""]

    # Table: main results per tier and arm.
    L += [r"\newcommand{\MainResultsTable}{%",
          r"\begin{table*}[t]\centering",
          r"\caption{Accuracy, cost and latency by task tier and system arm. Accuracy is exact agreement with the "
          r"independent reference within tolerance; 95\% Wilson intervals in brackets. Latency is wall-clock per answer; "
          r"tokens are LLM tokens per answer (prompt + completion, including hidden reasoning).}",
          r"\label{tab:main}\small\resizebox{\textwidth}{!}{%",
          r"\begin{tabular}{llrrrrrr}\toprule",
          r"Tier & Arm & $n$ & Exact acc.\ (\%) [95\% CI] & No answer (\%) & Median lat.\ (ms) & p95 lat.\ (ms) & Mean tokens \\ \midrule"]
    for tier in (1, 2, 3):
        first = True
        for arm in ("B_tool", "A_direct", "A_deployed"):
            key = f"T{tier}-{arm}"
            if f"{key}-n" not in vals:
                continue
            L.append(f"{TIER_LABEL[tier] if first else ''} & {ARM_LABEL[arm]} & {vals[key + '-n']} & "
                     f"{vals[key + '-acc']} [{vals[key + '-lo']}, {vals[key + '-hi']}] & {vals[key + '-noans']} & "
                     f"{vals[key + '-medlat']} & {vals[key + '-p95lat']} & {vals[key + '-tok']} \\\\")
            first = False
        if tier != 3:
            L.append(r"\midrule")
    L += [r"\bottomrule\end{tabular}}\end{table*}}", ""]

    # Table: per-case answers for Tier 2/3 (appendix).
    L += [r"\newcommand{\PerCaseTable}{%", r"\begin{table*}[t]\centering",
          r"\caption{Per-case answers on the time-series tiers (all repeats). ``--'' = no final numeric answer.}",
          r"\label{tab:percase}\scriptsize", r"\begin{tabular}{lrlll}\toprule",
          r"Case & Reference & Foleo tools & Groq + full data & Groq chatbot (no prices) \\ \midrule"]
    for case in dict.fromkeys(r["case"] for r in rows if r["tier"] in (2, 3)):
        cells = []
        for arm in ("B_tool", "A_direct", "A_deployed"):
            cells.append(", ".join("--" if r["parsed"] is None else f"{r['parsed']:g}"
                                   for r in rows if r["case"] == case and r["arm"] == arm))
        L.append(f"{tex(case)} & {truth[case]:g} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule\end{tabular}\end{table*}}", ""]

    # Table: external assistants.
    if ext_rows:
        L += [r"\newcommand{\ExternalTable}{%", r"\begin{table}[t]\centering",
              r"\caption{External assistants on the identical prompt pack (single session, built-in code execution "
              r"used by both). Exact answers / prompts.}",
              r"\label{tab:external}\small", r"\begin{tabular}{lrrrr}\toprule",
              r"Assistant & Tier 1 & Tier 2 & Tier 3 & Total \\ \midrule"]
        for model in dict.fromkeys(r["model"] for r in ext_rows):
            L.append(f"{EXTERNAL_LABEL.get(model, tex(model))} & " + " & ".join(vals.get(f"X-{model}-{t}", "--") for t in (1, 2, 3, "all")) + r" \\")
        L += [r"\bottomrule\end{tabular}\end{table}}", ""]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {args.out} (run complete={complete}, {len(vals)} values)")


if __name__ == "__main__":
    main()
