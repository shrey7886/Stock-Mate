import sys
import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR  = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "processed" / "tft_features_all_stocks.parquet"

sys.path.insert(0, str(ROOT_DIR / "data_pipeline"))
sys.path.insert(0, str(ROOT_DIR / "data_pipeline" / "ingestion"))

from fetch_stocks import START_DATE, END_DATE

def add_time_index(df):
    df = df.copy()
    df["time_idx"] = df.groupby("ticker").cumcount()
    return df

def time_based_split(df, train_frac=0.8):
    all_dates  = df["date"].sort_values().unique()
    cutoff_idx = int(len(all_dates) * train_frac)
    cutoff     = all_dates[cutoff_idx]

    train_df = df[df["date"] < cutoff].copy()
    test_df  = df[df["date"] >= cutoff].copy()

    ENCODER_LENGTH = 60
    context_dfs = []
    for ticker, grp in train_df.groupby("ticker"):
        context = grp.sort_values("date").tail(ENCODER_LENGTH)
        context_dfs.append(context)

    test_with_context = pd.concat(
        [pd.concat(context_dfs), test_df], ignore_index=True
    )
    test_with_context = test_with_context.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    return train_df, test_with_context, cutoff

def verify_no_leakage(train_df, test_df, cutoff):
    print("=== LEAKAGE CHECK ===")

    train_dates     = set(train_df["date"].unique())
    true_test_dates = set(test_df[test_df["date"] >= cutoff]["date"].unique())
    true_overlap    = true_test_dates.intersection(train_dates)

    if len(true_overlap) == 0:
        print("  ✓ No test dates found in training set")
    else:
        print(f"  ✗ LEAKAGE: {len(true_overlap)} test dates found in training set")
        print(f"    Sample: {sorted(true_overlap)[:5]}")

    train_max = train_df["date"].max()
    test_min  = test_df[test_df["date"] >= cutoff]["date"].min()
    print(f"  Train max date : {train_max.date()}")
    print(f"  Test min date  : {pd.Timestamp(test_min).date()}")
    print(f"  Cutoff         : {pd.Timestamp(cutoff).date()}")

    if train_max < test_min:
        print("  ✓ Train ends strictly before test begins")
    else:
        print("  ✗ LEAKAGE: Train and test dates overlap")

    leaky_tickers = []
    for ticker, grp in train_df.groupby("ticker"):
        if not grp["time_idx"].is_monotonic_increasing:
            leaky_tickers.append(ticker)

    if not leaky_tickers:
        print("  ✓ time_idx monotonically increasing for all tickers")
    else:
        print(f"  ✗ time_idx not monotonic for: {leaky_tickers}")

    print("=== END LEAKAGE CHECK ===\n")

def main():
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df = add_time_index(df)
    train_df, test_df, cutoff = time_based_split(df)
    verify_no_leakage(train_df, test_df, cutoff)

if __name__ == "__main__":
    main()