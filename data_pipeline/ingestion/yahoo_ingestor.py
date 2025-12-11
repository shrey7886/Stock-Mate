# data_pipeline/ingestion/yahoo_ingestor.py

import yfinance as yf
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_single_ticker(ticker: str, period="5y", interval="1d") -> pd.DataFrame:
    print(f"Fetching {ticker} ...")

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data found for {ticker}")

    df = df.reset_index()
    df["symbol"] = ticker
    df.rename(columns=str.lower, inplace=True)  # Date->date, Open->open

    return df


def save_ticker(df: pd.DataFrame, ticker: str):
    out_path = RAW_DIR / f"{ticker}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved raw data for {ticker} → {out_path}")


def fetch_universe(tickers: list[str]):
    for t in tickers:
        try:
            df = fetch_single_ticker(t)
            save_ticker(df, t)
        except Exception as e:
            print(f"Failed for {t}: {e}")


if __name__ == "__main__":
    TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
    fetch_universe(TICKERS)
