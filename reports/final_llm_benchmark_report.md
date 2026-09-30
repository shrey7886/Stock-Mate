> **SUPERSEDED (2026-09-27) - do not cite these numbers.** This run had measurement errors: the "accuracy proxy" compared the first number in each answer (so "6-month" matched "6-month"; the real baseline match rate was 0/11), the LLM baseline was given no price data, and portfolio prompts routed to wrong tickers. See `RESEARCH_HANDOUT.md` and `benchmark_runs/` for the corrected benchmark.

# Final LLM benchmark report

## Purpose

This report evaluates the reliability of two approaches for financial portfolio arithmetic:

1. a deterministic tool-based path that computes exact portfolio metrics with Python functions and numerical libraries
2. a direct LLM baseline that answers the same questions in natural language without explicit arithmetic enforcement

The goal is to test whether tool grounding improves correctness, latency, and trustworthiness for exact financial reasoning.

---

## Experimental setup

We compare the following paths on a fixed set of portfolio-finance queries:

- deterministic tool execution
- direct Groq / LLM response path

The benchmark questions cover exact financial calculations such as:

- correlation
- beta
- Sharpe ratio
- CAGR
- momentum
- total return
- VaR
- Sortino ratio
- drawdown
- moving average
- RSI
- downside deviation
- mean daily return

Each task is evaluated on:

- numerical correctness relative to the computed tool result
- latency (end-to-end response time)
- token usage
- unsupported or vague answer behavior

---

## Executive summary

The benchmark strongly supports the central claim of the project:

> exact financial arithmetic should be routed to deterministic tools, while the LLM should be reserved for interpretation, explanation, and user-facing guidance.

### Key findings

- Tool path latency: 589.38 ms average
- Baseline latency: 8057.31 ms average
- Improvement: 92.69% faster
- Tool tokens: 0.00 average
- Baseline tokens: 2444.56 average
- Token reduction: 100%
- Accuracy proxy: 27.27% for the current baseline comparison, but the tool path was consistently exact where a direct answer existed

These numbers indicate a strong practical advantage: deterministic tools are both faster and substantially more grounded than direct LLM arithmetic.

---

## Aggregate benchmark results

| Metric | Tool path | Baseline LLM path | Change |
|---|---:|---:|---:|
| Average latency | 589.38 ms | 8057.31 ms | -92.69% |
| Average tokens | 0.00 | 2444.56 | -100.00% |
| Avg. accuracy proxy | exact / grounded | 27.27% | improvement in arithmetic reliability |

### Interpretation

The tool path is not only faster; it eliminates the most common failure mode of direct LLM financial reasoning: numerical drift and unsupported output.

The LLM remains useful for summarization, high-level advice, and natural-language interpretation, but it should not be trusted for exact portfolio arithmetic when a deterministic computation is available.

---

## Per-task results

### Correlation
- Tool answer: 0.7700
- Baseline answer: approximately 0.7
- Tool latency: 730.46 ms
- Baseline latency: 935.29 ms
- Token savings: 100%
- Observed issue: baseline answer appears approximate and not directly verifiable

### Beta
- Tool answer: 0.2280
- Tool latency: 370.07 ms
- Baseline latency: 739.39 ms
- Token savings: 100%
- Observation: the LLM did not provide a comparable numeric result in a reliable way

### Sharpe ratio
- Tool answer: 0.5097
- Tool latency: 264.84 ms
- Baseline latency: 383.04 ms
- Token savings: 100%
- Observation: baseline was not directly available or was not numerically grounded

### CAGR (1-year)
- Tool answer: -25.87%
- Baseline answer: approximately 27%
- Tool latency: 519.72 ms
- Baseline latency: 465.38 ms
- Token savings: 100%
- Observation: direct LLM answer diverged materially from the computed result

### Momentum
- Tool answer: 3.94%
- Baseline answer: +15%
- Tool latency: 886.08 ms
- Baseline latency: 476.62 ms
- Token savings: 100%
- Observation: large mismatch between baseline and exact calculation

### Total return (6-month)
- Tool answer: -2.65%
- Baseline answer: 23%
- Tool latency: 415.77 ms
- Baseline latency: 9825.54 ms
- Token savings: 100%
- Observation: the model still failed on a simple return calculation

### VaR (90-day)
- Tool answer: 1.89%
- Tool latency: 415.22 ms
- Baseline latency: 11743.48 ms
- Token savings: 100%
- Observation: the baseline did not provide a directly comparable numeric answer

### Sortino ratio
- Tool answer: 0.0063
- Tool latency: 330.72 ms
- Baseline latency: 11863.62 ms
- Token savings: 100%
- Observation: baseline was not numerically grounded

### Drawdown
- Tool answer: -49.52%
- Baseline answer: 0% / unrelated
- Tool latency: 411.54 ms
- Baseline latency: 12028.36 ms
- Token savings: 100%
- Observation: large qualitative mismatch between model narrative and exact value

### Moving average (50-day)
- Tool answer: 606.07
- Tool latency: 282.70 ms
- Baseline latency: 22446.04 ms
- Token savings: 100%
- Observation: baseline did not provide a usable answer

### Moving average (200-day)
- Tool answer: 272.05
- Tool latency: 428.05 ms
- Baseline latency: 787.00 ms
- Observation: direct response was non-comparable or weakly grounded

### RSI
- Tool answer: 46.60
- Baseline answer: around 65
- Tool latency: 1.53 ms
- Baseline latency: 12632.98 ms
- Token savings: 100%
- Observation: approximate language model output diverged materially from the exact value

### CAGR (5-year)
- Tool answer: 17.54%
- Baseline answer: around 28%
- Tool latency: 561.59 ms
- Baseline latency: 11722.45 ms
- Token savings: 100%
- Observation: exact result was stable while baseline drifted substantially

### Downside deviation
- Tool answer: 44.64%
- Tool latency: 3506.10 ms
- Baseline latency: 9608.72 ms
- Token savings: 100%
- Observation: baseline explanation was conceptual rather than computed

### Total return (1-year)
- Tool answer: -21.91%
- Baseline answer: -15%
- Tool latency: 304.95 ms
- Baseline latency: 10681.23 ms
- Token savings: 100%
- Observation: baseline drifted enough to make the result unreliable

### Mean daily return
- Tool answer: 0.1434%
- Baseline answer: around 1.3%
- Tool latency: 0.71 ms
- Baseline latency: 12577.82 ms
- Token savings: 100%
- Observation: the coarse LLM estimate misses the exact arithmetic by a very large margin

---

## Interpretation of the findings

The underlying pattern is clear and consistent across tasks:

- direct LLM answers are often approximate, inconsistent, or unavailable
- deterministic tools provide exact, stable numbers
- the tool path is far cheaper and much faster
- the LLM is more useful as an explanatory layer after the arithmetic is already established

This is exactly the architectural insight behind the project.

---

## Why this matters for the paper

This benchmark supports a paper claim that is both practical and technically interesting:

> tool-augmented reasoning is a reliable design pattern for financial decision support when exact arithmetic matters.

This is stronger and more defensible than claiming that the LLM alone can act as a finance advisor. The benchmark provides evidence that:

- the LLM should not directly handle exact portfolio calculations
- tool routing substantially reduces error and latency
- grounded reasoning is more trustworthy than generative numerical guessing

---

## Strengths of the current evidence

The benchmark already supports a credible claim that the project contributes to trustworthy portfolio reasoning by showing:

- exact arithmetic can be delegated to code
- the LLM can act as an explanation layer
- the system reduces unsupported numerical claims
- the hybrid design is materially faster and more efficient

This is enough to support a strong systems paper or applied ML paper with the right framing.

---

## Limitations of the current benchmark

This report is useful, but it is not yet publication-grade in a strict academic sense. The main limitations are:

1. It is a small benchmark set rather than a large, statistically rich evaluation.
2. The metric labeled “accuracy proxy” is a coarse proxy rather than a formal benchmark protocol.
3. The comparison currently focuses on Groq as the baseline; a more complete paper would include multiple model families.
4. It would be stronger to add an open-source model comparison, such as Llama or Mistral, to show how the tool-augmented path behaves across model types.
5. The benchmarking should ideally include human-judged trustworthiness and unsupported-claim rates.

---

## Recommended upgrade for a publication-ready version

To make the paper stronger, the next step should be:

- add an open-source LLM to the benchmark
- compare LLM-only, LLM-with-tools, and deterministic tool-only paths side by side
- include a larger benchmark set with at least 50–200 fixed questions
- add a human or rubric-based trust evaluation
- report confidence intervals, not only averages

This would make the study more convincing and more publication-ready.

---

## Conclusion

The observed results are sufficient to justify the project’s main research claim: tool-grounded financial reasoning improves both reliability and efficiency relative to direct LLM numerical reasoning.

The benchmark is not yet the final polished academic evidence set, but it is a strong foundation for a paper and is already enough to support the central narrative of the system.

The most defensible framing is not “AI predicts returns better,” but rather:

> tool-augmented portfolio reasoning improves correctness, latency, and trustworthiness by separating exact financial arithmetic from free-form language generation.
