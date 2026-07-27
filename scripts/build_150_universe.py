"""
scripts/build_150_universe.py
─────────────────────────────
Fetches OHLCV data from Yahoo Finance for 150 stocks, applies all feature-
engineering steps (technical indicators, volatility, calendar, targets), and
writes:
  - data/raw/{TICKER}.parquet          (raw OHLCV, skipped if already present)
  - data/processed/{TICKER}_tft.parquet (feature-engineered, always regenerated)
  - data/processed/patchtst_150_universe.parquet (combined panel for PatchTST)

Usage:
    python scripts/build_150_universe.py [--force-raw] [--out NAME]

    --force-raw : re-download raw data even if already cached
    --out       : output parquet name (default: patchtst_150_universe)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from data_pipeline.ingestion.yahoo_ingestor import fetch_single_ticker, save_ticker
from ml_service.data.dataset_generator import generate_tft_dataset_for_ticker

# ─────────────────────────────────────────────────────────────────
#  150-STOCK UNIVERSE
# ─────────────────────────────────────────────────────────────────
UNIVERSE_150: list[str] = [
    # ── Technology (27) ──────────────────────────────────────────
    "AAPL", "MSFT", "NVDA", "AMD", "META", "GOOGL", "ORCL", "CRM", "INTC",
    "AMZN", "TSLA",
    "ADBE", "NOW", "AVGO", "QCOM", "TXN", "MU", "CSCO", "IBM", "ACN",
    "NFLX", "UBER", "PLTR", "CRWD", "DDOG", "PANW", "INTU",

    # ── Financials (17) ──────────────────────────────────────────
    "JPM", "BAC", "GS", "MS", "WFC", "C", "AXP",
    "V", "MA", "BLK", "SCHW", "USB", "PNC", "COF", "SPGI", "MCO", "ICE",

    # ── Energy (9) ───────────────────────────────────────────────
    "XOM", "CVX", "BP", "SLB", "COP",
    "OXY", "PSX", "VLO", "NEE",

    # ── Consumer Staples (13) ────────────────────────────────────
    "PG", "KO", "PEP", "WMT", "COST", "MCD",
    "CL", "PM", "MO", "MDLZ", "GIS", "KR", "SYY",

    # ── Healthcare (17) ──────────────────────────────────────────
    "JNJ", "PFE", "MRK", "ABBV", "UNH",
    "BMY", "GILD", "AMGN", "TMO", "DHR", "SYK", "MDT", "ABT", "BSX", "CVS",
    "ISRG", "VRTX",

    # ── Industrials (16) ─────────────────────────────────────────
    "CAT", "DE", "GE", "HON", "MMM",
    "RTX", "LMT", "BA", "NOC", "UPS", "FDX",
    "LIN", "APD", "SHW", "EMR", "ETN",

    # ── Consumer Discretionary (13) ──────────────────────────────
    "NKE", "SBUX", "TGT", "HD", "LOW", "F", "GM",
    "TJX", "BKNG", "ABNB", "DIS", "LULU", "CMG",

    # ── Communication Services (5) ───────────────────────────────
    "T", "VZ", "CMCSA", "CHTR", "TMUS",

    # ── Materials (5) ────────────────────────────────────────────
    "NEM", "FCX", "DD", "NUE", "ECL",

    # ── Real Estate (5) ──────────────────────────────────────────
    "AMT", "PLD", "EQIX", "SPG", "O",

    # ── Utilities (5) ────────────────────────────────────────────
    "DUK", "SO", "AEP", "EXC", "D",

    # ── Other / Diversified (18) ─────────────────────────────────
    "ZTS", "REGN", "HCA", "CI", "ELV",
    "BX", "KKR", "CARR", "OTIS", "CTAS", "WM", "RSG",
    "ADSK", "MELI", "NET", "SNOW", "PYPL", "SQ",
]

# Deduplicate while preserving order
seen: set[str] = set()
UNIVERSE_150_DEDUP: list[str] = []
for t in UNIVERSE_150:
    if t not in seen:
        seen.add(t)
        UNIVERSE_150_DEDUP.append(t)

# Trim / pad to exactly 150
UNIVERSE_150_DEDUP = UNIVERSE_150_DEDUP[:150]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def fetch_missing_raw(tickers: list[str], force: bool = False) -> None:
    """Download OHLCV from Yahoo Finance for any ticker whose raw file is absent."""
    print(f"\n{'─'*60}")
    print(f" Stage 1 ─ Fetching raw OHLCV data ({len(tickers)} tickers)")
    print(f"{'─'*60}")

    for ticker in tickers:
        raw_path = RAW_DIR / f"{ticker}.parquet"
        if raw_path.exists() and not force:
            print(f"  [SKIP]  {ticker}  (already cached)")
            continue
        try:
            df = fetch_single_ticker(ticker, period="10y", interval="1d")
            save_ticker(df, ticker)
        except Exception as exc:
            print(f"  [FAIL]  {ticker}: {exc}")


def process_all(tickers: list[str]) -> pd.DataFrame:
    """Run feature engineering for every ticker and return the combined frame."""
    print(f"\n{'─'*60}")
    print(f" Stage 2 ─ Feature engineering ({len(tickers)} tickers)")
    print(f"{'─'*60}")

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for ticker in tickers:
        try:
            df = generate_tft_dataset_for_ticker(ticker)
            frames.append(df)
        except Exception as exc:
            print(f"  [FAIL]  {ticker}: {exc}")
            failed.append(ticker)

    if not frames:
        raise ValueError("No datasets could be generated.")

    if failed:
        print(f"\n  Skipped {len(failed)} ticker(s): {', '.join(failed)}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Combined shape: {combined.shape}")
    print(f"  Symbols ({combined['symbol'].nunique()}): {sorted(combined['symbol'].unique().tolist())}")
    return combined


def save_universe(df: pd.DataFrame, name: str) -> Path:
    print(f"\n{'─'*60}")
    print(f" Stage 3 ─ Saving combined universe")
    print(f"{'─'*60}")

    out_path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"  Saved → {out_path}  ({size_mb:.1f} MB,  {len(df):,} rows)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 150-stock PatchTST dataset.")
    parser.add_argument(
        "--force-raw",
        action="store_true",
        help="Re-download raw OHLCV even if already cached.",
    )
    parser.add_argument(
        "--out",
        default="patchtst_150_universe",
        help="Output parquet filename (without .parquet extension).",
    )
    args = parser.parse_args()

    tickers = UNIVERSE_150_DEDUP
    print(f"\n[build_150_universe] Universe: {len(tickers)} stocks")

    fetch_missing_raw(tickers, force=args.force_raw)
    combined = process_all(tickers)
    out_path = save_universe(combined, args.out)

    print(f"\n[DONE]  Universe parquet ready at: {out_path}")
    print(
        f"        Train PatchTST with:\n"
        f"        python ml_service/training/train_patchtst.py \\\n"
        f"          --data data/processed/{args.out}.parquet \\\n"
        f"          --model-dir ml_service/models/saved_models/patchtst_150 \\\n"
        f"          --max-steps 2000"
    )


if __name__ == "__main__":
    main()
