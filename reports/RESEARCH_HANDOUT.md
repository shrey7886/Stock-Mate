# Foleo (Stock-Mate) Research Handout — corrected 2026-09-27

> The previous version of this handout, `final_llm_benchmark_report.md` and `quantitative_benchmark_report.md`
> contained measurement errors (see §5). Do not cite numbers from them. The paper draft is in `../paper/`.

## 1) Research question
When a portfolio assistant is asked for exact numbers (P&L, returns, Sharpe, VaR, drawdown, RSI), should the LLM
compute them, or should deterministic tools? Foleo's design: the LLM handles conversation; a rule-based router
sends every quantitative request — including implicit ones like "what is my Sharpe ratio?" — to 16 tested functions
that answer with 0 LLM tokens.

## 2) Where the code is
- `llm_orchestrator/utils/quantitative_tools.py` — router, ticker resolution (`IDEA` + NSE → `IDEA.NS`), portfolio-level
  metrics (value series Σ qty × close), metric functions.
- `llm_orchestrator/agents/response_agent.py` — LLM layer (Groq `openai/gpt-oss-120b`).
- `scripts/run_research_benchmark.py` — the official benchmark (resumable).
- `scripts/score_external.py` — scores ChatGPT/Claude answers against the same ground truth.
- `scripts/make_paper_tables.py` — generates every number in the paper from the raw logs.

## 3) Benchmark design (run `reports/benchmark_runs/2026-09-27_1546/`)
- **Same data for every system:** pinned NSE daily closes (25 Jun–25 Sep 2026, 67 rows), holdings in Upstox schema.
- **Tiers:** 1 = portfolio arithmetic (10 prompts), 2 = stock time-series metrics (12), 3 = implicit portfolio
  risk metrics (4).
- **Arms:** Foleo tools; Groq with full data (temp 0); Groq chatbot as deployed (holdings, no prices); ChatGPT and
  Claude via the external prompt pack.
- **Ground truth:** independent pandas implementation; every prompt states the formula; only the `FINAL:` line
  is scored; Wilson 95% CIs; quota errors are logged, never scored.

## 4) Final findings (run complete 2026-09-28; full tables in `summary.md`)
1. **Tools:** 26/26 exact, 0 tokens, 0.1–3.3 ms median compute; routing correct on 26/26 incl. implicit "my portfolio".
2. **LLM on simple arithmetic is reliable** (Tier 1: 30/30 with full data, 29/30 via the deployed chatbot). The
   case for tools is *not* "LLMs can't add".
3. **LLM with full data on time-series metrics:** Tier 2 21/36, Tier 3 3/12; ~3,900–4,900 tokens and ~8–10 s per
   answer. All 23 missing answers were token-budget exhaustion (VaR 0/9; portfolio Sharpe/VaR/drawdown 0/9).
   One silent arithmetic error (HDFCBANK RSI 62.68 vs 61.40). Only 6/12 Tier-2 cases gave identical answers across repeats.
4. **Without price data the deployed chatbot** gave no value in 47/48 requests, but once reported the gain since
   purchase (−6.74%) as the "three-month return" (true −19.27%).
5. **External assistants used code execution:** Claude 26/26, ChatGPT 23/26 — ChatGPT's RSI answers used a different
   formula than the one specified. Code removes arithmetic error, not definition error.

## 5) Bugs found and fixed (all with regression tests)
- Chatbot returned **empty replies** 18/18 times: gpt-oss spent the 260-token budget on hidden reasoning.
- **Ticker mis-resolution:** "portfolio"/"this"/"maximum" treated as tickers; `IDEA` resolved to a US penny stock;
  "portfolio Sharpe" used only the first holding. 4 of the 6 old benchmark prompts were actually errors.
- **Beta** mixed sample covariance with population variance; "var" matched inside words; default periods overridden.
- **Old accuracy proxy** compared the first number in each answer ("6-month" = "6-month"): 3/11 false matches,
  true agreement 0/11. The old "68% faster" figure averaged failed tool calls with two LLM outliers.

## 6) What the paper can and cannot claim
- **Can:** tool grounding gives exact, complete, zero-token, definition-stable and auditable answers; the LLM is
  accurate on simple arithmetic but costly/brittle on longer computations.
- **Cannot:** "LLMs hallucinate portfolio arithmetic" in general; statistical claims beyond one model, 26 prompts and
  3 repeats; anything about investment performance (the Buy/Hold/Trim/Exit rule is a test rule, not advice).
