# ml_service/data/dataset_generator.py

from pathlib import Path
import pandas as pd

from ml_service.data.feature_functions import (
    add_price_features,
    add_technical_indicators,
    add_volatility_features,
    add_sentiment_placeholders,
    add_calendar_features,
    add_targets
)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes Yahoo Finance parquet files that contain either:
    - MultiIndex columns
    - Stringified tuple column names like "('close', 'aapl')"
    - Uppercase/lowercase inconsistencies
    """

    new_cols = []

    for col in df.columns:
        col_str = str(col)

        # Case 1: Stringified tuple → "('close', 'aapl')"
        if col_str.startswith("(") and "," in col_str:
            # Extract first element
            cleaned = col_str.split(",")[0]          # "('close'"
            cleaned = cleaned.replace("('", "")      # "close"
            cleaned = cleaned.replace("'", "")       # remove quotes
            new_cols.append(cleaned.lower())
            continue

        # Case 2: Normal string column
        new_cols.append(col_str.lower())

    df.columns = new_cols
    return df


def ensure_close_column(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Ensures the dataframe always contains a usable 'close' column.
    """

    if "close" not in df.columns:
        if "adj close" in df.columns:
            df["close"] = df["adj close"]
        else:
            raise KeyError(
                f"'close' column missing for {ticker}. Columns: {df.columns}"
            )

    return df


def generate_tft_dataset_for_ticker(ticker: str) -> pd.DataFrame:
    raw_path = RAW_DIR / f"{ticker}.parquet"

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file missing: {raw_path}")

    df = pd.read_parquet(raw_path)

    # -------------- FIX COLUMN ISSUES --------------
    df = clean_columns(df)
    df = ensure_close_column(df, ticker)

    # -------------- ADD TIME INDEX --------------
    df["time_idx"] = range(len(df))

    # -------------- APPLY FEATURE ENGINEERING --------------
    df = add_price_features(df)
    df = add_technical_indicators(df)
    df = add_volatility_features(df)
    df = add_sentiment_placeholders(df)
    df = add_calendar_features(df)
    df = add_targets(df)

    # -------------- CLEAN & SAVE --------------
    df = df.dropna().reset_index(drop=True)

    out_path = PROCESSED_DIR / f"{ticker}_tft.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[OK] Saved TFT dataset → {out_path}")

    return df


def build_universe_dataset(tickers: list[str]) -> pd.DataFrame:
    all_dfs = []

    for t in tickers:
        try:
            df = generate_tft_dataset_for_ticker(t)
            all_dfs.append(df)
        except Exception as e:
            print(f"[SKIP] {t}: {e}")

    if not all_dfs:
        raise ValueError("No datasets generated. Cannot concatenate.")

    full = pd.concat(all_dfs, ignore_index=True)

    out_path = PROCESSED_DIR / "tft_full_universe.parquet"
    full.to_parquet(out_path, index=False)

    print(f"\n[OK] Saved full universe dataset → {out_path}\n")
    return full


if __name__ == "__main__":
    TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
    build_universe_dataset(TICKERS)
