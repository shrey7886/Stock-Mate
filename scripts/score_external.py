"""Score manually recorded ChatGPT / Claude answers with the same rules as run_research_benchmark.py.

Fill <run>/external_results_template.csv (final_answer = what the FINAL line said, seconds = wall clock),
then: python scripts/score_external.py reports/benchmark_runs/<run>/external_results_template.csv
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

TOLERANCE = {"rsi14": 0.1, "sharpe": 0.005}  # everything else: 0.011 (Tier 1 rupees/%), 0.02 (Tier 2/3 %)


def parse(answer: str, truth):
    answer = (answer or "").replace("−", "-").replace("₹", "").replace(",", "")
    if isinstance(truth, str):
        hits = [lab for lab in ("Buy", "Hold", "Trim", "Exit") if lab.lower() in answer.lower()]
        return hits[0] if len(hits) == 1 else None
    m = re.search(r"-?\d+(?:\.\d+)?", answer)
    return float(m.group()) if m else None


def main(path: str) -> None:
    sheet = Path(path)
    truth = json.loads((sheet.parent / "ground_truth.json").read_text())
    stats = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, []]))  # model -> tier -> [n, correct, no_answer, secs]
    rows = [r for r in csv.DictReader(sheet.open(encoding="utf-8")) if r["final_answer"].strip()]
    for r in rows:
        case, expected = r["case"], truth[r["case"]]
        metric = case.split(":")[1]
        tier = 1 if case.startswith(("upstox_snapshot", "realistic_8")) else 3 if case.startswith("portfolio") else 2
        tol = TOLERANCE.get(metric, 0.011 if tier == 1 else 0.02)
        got = parse(r["final_answer"], expected)
        ok = got is not None and (got == expected if isinstance(expected, str) else abs(got - expected) <= tol)
        s = stats[r["model"]][tier]
        s[0] += 1
        s[1] += ok
        s[2] += got is None
        if r["seconds"].strip():
            s[3].append(float(r["seconds"]))
        print(f"{r['model']:8} {case:32} truth={expected!s:>10} got={got!s:>10} {'OK' if ok else 'WRONG'}")

    print("\n| Model | Tier | n | Exact acc % | No answer % | Median seconds |\n|---|---:|---:|---:|---:|---:|")
    for model, tiers in stats.items():
        for tier, (n, k, none, secs) in sorted(tiers.items()):
            med = sorted(secs)[len(secs) // 2] if secs else "n/a"
            print(f"| {model} | {tier} | {n} | {100 * k / n:.1f} | {100 * none / n:.1f} | {med} |")


if __name__ == "__main__":
    main(sys.argv[1])
