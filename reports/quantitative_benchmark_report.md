# Quantitative benchmark report

## Summary

This report compares the deterministic tool path against the baseline Groq response path for numeric finance questions.

### Aggregate results
- Average tool latency: 589.38 ms
- Average baseline latency: 8057.31 ms
- Latency change vs baseline: -92.69% (tool path is about 92.7% faster)
- Average tool tokens: 0.00
- Average baseline tokens: 2444.56
- Token change vs baseline: -100.00%
- Average accuracy proxy: 27.27%

## Per-tool breakdown

### Correlation
- Tool answer: 0.7700
- Baseline answer: around 0.7
- Tool latency: 730.46 ms
- Baseline latency: 935.29 ms
- Token savings: 100.0%
- Accuracy proxy: 0.0%

### Beta
- Tool answer: 0.2280
- Baseline answer: not directly comparable
- Tool latency: 370.07 ms
- Baseline latency: 739.39 ms
- Token savings: 100.0%
- Accuracy proxy: n/a

### Sharpe ratio
- Tool answer: 0.5097
- Baseline answer: not directly available
- Tool latency: 264.84 ms
- Baseline latency: 383.04 ms
- Token savings: 100.0%
- Accuracy proxy: n/a

### CAGR (1-year)
- Tool answer: -25.87%
- Baseline answer: around 27%
- Tool latency: 519.72 ms
- Baseline latency: 465.38 ms
- Token savings: 100.0%
- Accuracy proxy: 0.0%

### Momentum
- Tool answer: 3.94%
- Baseline answer: +15%
- Tool latency: 886.08 ms
- Baseline latency: 476.62 ms
- Token savings: 100.0%
- Accuracy proxy: 0.0%

### Total return (6-month)
- Tool answer: -2.65%
- Baseline answer: 23%
- Tool latency: 415.77 ms
- Baseline latency: 9825.54 ms
- Token savings: 100.0%
- Accuracy proxy: 100.0%

### VaR (90-day)
- Tool answer: 1.89%
- Baseline answer: not directly available
- Tool latency: 415.22 ms
- Baseline latency: 11743.48 ms
- Token savings: 100.0%
- Accuracy proxy: 0.0%

### Sortino ratio
- Tool answer: 0.0063
- Baseline answer: not directly available
- Tool latency: 330.72 ms
- Baseline latency: 11863.62 ms
- Token savings: 100.0%
- Accuracy proxy: n/a

### Drawdown
- Tool answer: -49.52%
- Baseline answer: 0% / unrelated
- Tool latency: 411.54 ms
- Baseline latency: 12028.36 ms
- Token savings: 100.0%
- Accuracy proxy: 0.0%

### Moving average (50-day)
- Tool answer: 606.07
- Baseline answer: not available
- Tool latency: 282.70 ms
- Baseline latency: 22446.04 ms
- Token savings: 100.0%
- Accuracy proxy: 100.0%

### Moving average (200-day)
- Tool answer: 272.05
- Baseline answer: not relevant
- Tool latency: 428.05 ms
- Baseline latency: 787.00 ms
- Token savings: 100.0%
- Accuracy proxy: n/a

### RSI
- Tool answer: 46.60
- Baseline answer: around 65
- Tool latency: 1.53 ms
- Baseline latency: 12632.98 ms
- Token savings: 100.0%
- Accuracy proxy: 0.0%

### CAGR (5-year)
- Tool answer: 17.54%
- Baseline answer: around 28%
- Tool latency: 561.59 ms
- Baseline latency: 11722.45 ms
- Token savings: 100.0%
- Accuracy proxy: 100.0%

### Downside deviation
- Tool answer: 44.64%
- Baseline answer: conceptual explanation only
- Tool latency: 3506.10 ms
- Baseline latency: 9608.72 ms
- Token savings: 100.0%
- Accuracy proxy: n/a

### Total return (1-year)
- Tool answer: -21.91%
- Baseline answer: -15%
- Tool latency: 304.95 ms
- Baseline latency: 10681.23 ms
- Token savings: 100.0%
- Accuracy proxy: 0.0%

### Mean daily return
- Tool answer: 0.1434%
- Baseline answer: around 1.3%
- Tool latency: 0.71 ms
- Baseline latency: 12577.82 ms
- Token savings: 100.0%
- Accuracy proxy: 0.0%

## Cleanup summary

Removed unused TFT and transformer-related artifacts including:
- training logs and experiment logs
- Lightning checkpoint directories
- TFT model checkpoints
- TFT parquet datasets
- TFT training and evaluation scripts
- old TFT feature-importance outputs

This leaves the active project files intact while removing the noisy experimental artifacts.
