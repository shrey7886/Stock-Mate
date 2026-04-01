import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT_DIR  = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "processed" / "tft_features_all_stocks.parquet"
MODEL_DIR = ROOT_DIR / "ml_service" / "models" / "tft_trained"

sys.path.insert(0, str(ROOT_DIR / "ml_service" / "training"))

def main():
    from train_tft import (
        load_and_clean, add_time_index, time_based_split,
        build_datasets, BATCH_SIZE
    )
    from pytorch_forecasting import TemporalFusionTransformer

    print("Loading data...")
    df = load_and_clean(DATA_PATH)
    df = add_time_index(df)
    train_df, test_df, cutoff = time_based_split(df, train_frac=0.8)
    train_dataset, val_dataset = build_datasets(train_df, test_df)

    # ── Load best model ───────────────────────────────────────────────────────
    ckpts = sorted(MODEL_DIR.glob("*.ckpt"))
    if not ckpts:
        print("[ERROR] No checkpoint found")
        return

    best_ckpt = str(ckpts[-1])
    print(f"Loading model: {best_ckpt}")
    model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
    model.eval()

    val_loader = val_dataset.to_dataloader(
        train=False, batch_size=BATCH_SIZE * 2,
        num_workers=0, shuffle=False
    )

    print("Running predictions...")
    predictions = model.predict(
        val_loader,
        mode         = "quantiles",
        return_index = True,
    )

    pred_index  = predictions.index.copy()
    pred_median = predictions.output[:, 0, 3].numpy()
    pred_index["predicted"] = pred_median

    true_test = test_df[test_df["date"] >= pd.Timestamp(cutoff)].copy()

    # ── Per ticker breakdown ──────────────────────────────────────────────────
    print("\n=== PER TICKER DIRECTIONAL ACCURACY ===")
    results = []

    for ticker in sorted(pred_index["ticker"].unique()):
        mask    = pred_index["ticker"] == ticker
        preds   = pred_median[mask]
        actuals = true_test[true_test["ticker"] == ticker]["returns_pct"].values

        n = min(len(preds), len(actuals))
        if n == 0:
            continue

        acc     = (np.sign(preds[:n]) == np.sign(actuals[:n])).mean()
        mae     = np.abs(preds[:n] - actuals[:n]).mean()
        results.append({
            "ticker":        ticker,
            "dir_acc":       acc,
            "mae":           mae,
            "n_predictions": n
        })

    results_df = pd.DataFrame(results).sort_values("dir_acc", ascending=False)

    print(results_df.to_string(index=False))
    print(f"\n  Mean dir accuracy  : {results_df['dir_acc'].mean():.2%}")
    print(f"  Median dir accuracy: {results_df['dir_acc'].median():.2%}")
    print(f"  Tickers > 55%      : {(results_df['dir_acc'] > 0.55).sum()}")
    print(f"  Tickers > 52%      : {(results_df['dir_acc'] > 0.52).sum()}")
    print(f"  Tickers < 50%      : {(results_df['dir_acc'] < 0.50).sum()}")
    print(f"  Mean MAE           : {results_df['mae'].mean():.6f}")

    out = MODEL_DIR / "per_ticker_accuracy.csv"
    results_df.to_csv(out, index=False)
    print(f"\n  Saved → {out}")
    print("=== END PER TICKER ===\n")

if __name__ == "__main__":
    main()