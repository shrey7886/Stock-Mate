# Benchmark summary (2026-09-27T15:46)

Model: `openai/gpt-oss-120b` via Groq. Repeats per LLM case: 3. Ground truth: independent pandas reference implementation.

## Tier 1 - portfolio arithmetic (holdings only)

| Arm | n | Exact acc % (95% CI) | Correct value anywhere % | No final answer % | Median abs err | Latency median / mean / p95 ms | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| B_tool | 10 | 100.0 (72.2-100.0) | 100.0 | 0.0 | 0.0 | 0.12 / 0.25 / 1.41 | 0 |
| A_deployed | 30 | 96.7 (83.3-99.4) | 100.0 | 3.3 | 0.0 | 1644.37 / 2289.88 / 3823.57 | 2694 |
| A_direct | 30 | 100.0 (88.6-100.0) | 100.0 | 0.0 | 0.0 | 2481.47 / 4299.17 / 7297.84 | 1421 |

Same answer on every repeat (cases): A_deployed 9/10, A_direct 10/10
Tool routed to the correct tool/scope: 10/10

| Case | Truth | B_tool | A_deployed | A_direct |
|---|---:|---|---|---|
| upstox_snapshot:total_invested | 89 | 89.0 | 89.0, 89.0, 89.0 | 89.0, 89.0, 89.0 |
| upstox_snapshot:total_current_value | 83 | 83.0 | 83.0, 83.0, 83.0 | 83.0, 83.0, 83.0 |
| upstox_snapshot:total_pnl | -6 | -6.0 | -6.0, -6.0, -6.0 | -6.0, -6.0, -6.0 |
| upstox_snapshot:total_pnl_pct | -6.74 | -6.74 | -6.74, -6.74, -6.74 | -6.74, -6.74, -6.74 |
| upstox_snapshot:decision | Trim | Trim | Trim, Trim, Trim | Trim, Trim, Trim |
| realistic_8:total_invested | 609742.2 | 609742.2 | 609742.2, 609742.2, 609742.2 | 609742.2, 609742.2, 609742.2 |
| realistic_8:total_current_value | 685857.65 | 685857.65 | 685857.65, 685857.65, 685857.65 | 685857.65, 685857.65, 685857.65 |
| realistic_8:total_pnl | 76115.45 | 76115.45 | 76115.45, 76115.45, 76115.45 | 76115.45, 76115.45, 76115.45 |
| realistic_8:total_pnl_pct | 12.48 | 12.48 | 12.48, -, 12.48 | 12.48, 12.48, 12.48 |
| realistic_8:decision | Buy | Buy | Buy, Buy, Buy | Buy, Buy, Buy |

## Tier 2 - single-stock time-series metrics (pinned 3-month prices)

| Arm | n | Exact acc % (95% CI) | Correct value anywhere % | No final answer % | Median abs err | Latency median / mean / p95 ms | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| B_tool | 12 | 100.0 (75.7-100.0) | 100.0 | 0.0 | 0.0 | 0.86 / 0.99 / 2.49 | 0 |
| A_deployed | 36 | 0.0 (0.0-9.6) | 0.0 | 100.0 | n/a | 1245.88 / 3128.40 / 7539.68 | 2519 |
| A_direct | 36 | 58.3 (42.2-72.9) | 58.3 | 38.9 | 0.0 | 8236.88 / 7630.09 / 14683.38 | 3882 |

Same answer on every repeat (cases): A_deployed 12/12, A_direct 6/12
Tool routed to the correct tool/scope: 12/12

| Case | Truth | B_tool | A_deployed | A_direct |
|---|---:|---|---|---|
| RELIANCE.NS:total_return | -6.99 | -6.99 | -, -, - | -6.99, -6.99, -6.99 |
| RELIANCE.NS:max_drawdown | -8.66 | -8.66 | -, -, - | -8.66, -8.66, -8.65 |
| RELIANCE.NS:var95 | 1.69 | 1.69 | -, -, - | -, -, - |
| RELIANCE.NS:rsi14 | 25.66 | 25.66 | -, -, - | 25.65, -, - |
| TCS.NS:total_return | -0.61 | -0.61 | -, -, - | -0.61, -0.61, -0.61 |
| TCS.NS:max_drawdown | -15.83 | -15.83 | -, -, - | -, -15.85, -15.84 |
| TCS.NS:var95 | 2.75 | 2.75 | -, -, - | -, -, - |
| TCS.NS:rsi14 | 22.21 | 22.21 | -, -, - | 22.22, -, 22.22 |
| HDFCBANK.NS:total_return | -7.62 | -7.62 | -, -, - | -7.62, -7.62, -7.62 |
| HDFCBANK.NS:max_drawdown | -17.2 | -17.2 | -, -, - | -, -17.2, -17.2 |
| HDFCBANK.NS:var95 | 2.19 | 2.19 | -, -, - | -, -, - |
| HDFCBANK.NS:rsi14 | 61.4 | 61.4 | -, -, - | 61.38, 62.68, 61.4 |

## Tier 3 - portfolio-level metrics, implicit 'my portfolio' question

| Arm | n | Exact acc % (95% CI) | Correct value anywhere % | No final answer % | Median abs err | Latency median / mean / p95 ms | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| B_tool | 4 | 100.0 (51.0-100.0) | 100.0 | 0.0 | 0.0 | 3.30 / 3.57 / 5.05 | 0 |
| A_deployed | 12 | 0.0 (0.0-24.3) | 0.0 | 91.7 | 12.53 | 1917.77 / 3396.85 / 18553.17 | 2285 |
| A_direct | 12 | 25.0 (8.9-53.2) | 25.0 | 75.0 | 0.0 | 9715.30 / 8970.09 / 13575.89 | 4867 |

Same answer on every repeat (cases): A_deployed 3/4, A_direct 4/4
Tool routed to the correct tool/scope: 4/4

| Case | Truth | B_tool | A_deployed | A_direct |
|---|---:|---|---|---|
| portfolio:sharpe | -4.2766 | -4.2766 | -, -, - | -, -, - |
| portfolio:var95 | 2.38 | 2.38 | -, -, - | -, -, - |
| portfolio:max_drawdown | -20.77 | -20.77 | -, -, - | -, -, - |
| portfolio:total_return | -19.27 | -19.27 | -6.74, -, - | -19.27, -19.27, -19.27 |

Price snapshot [first, last, n]: {"RELIANCE.NS": ["2026-06-25", "2026-09-25", 67], "TCS.NS": ["2026-06-25", "2026-09-25", 67], "HDFCBANK.NS": ["2026-06-25", "2026-09-25", 67], "IDEA.NS": ["2026-06-25", "2026-09-25", 67], "YESBANK.NS": ["2026-06-25", "2026-09-25", 67], "SUZLON.NS": ["2026-06-25", "2026-09-25", 67]}
