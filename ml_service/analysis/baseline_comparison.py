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
    # ── Rebuild datasets the same way as training ─────────────────────────────
    from train_tft import (
        load_and_clean, add_time_index, time_based_split,
        build_datasets, BATCH_SIZE, ENCODER_LENGTH, PREDICTION_LENGTH
    )
    from pytorch_forecasting import TemporalFusionTransformer
    from sklearn.metrics import mean_absolute_error

    print("Loading data...")
    df = load_and_clean(DATA_PATH)
    df = add_time_index(df)
    train_df, test_df, cutoff = time_based_split(df, train_frac=0.8)
    train_dataset, val_dataset = build_datasets(train_df, test_df)

    # ── Load best model ───────────────────────────────────────────────────────
    ckpts = sorted(MODEL_DIR.glob("*.ckpt"))
    if not ckpts:
        print("[ERROR] No checkpoint found in", MODEL_DIR)
        return

    best_ckpt = str(ckpts[-1])
    print(f"Loading model: {best_ckpt}")
    model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
    model.eval()

    # ── Get predictions ───────────────────────────────────────────────────────
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

    pred_median = predictions.output[:, 0, 3].numpy()  # day+1 median quantile

    # ── Get actual returns for test period ────────────────────────────────────
    true_test   = test_df[test_df["date"] >= pd.Timestamp(cutoff)].copy()
    true_test   = true_test.sort_values(["ticker", "date"])
    actual_vals = true_test["returns_pct"].values[:len(pred_median)]

    # ── Naive baselines ───────────────────────────────────────────────────────
    naive_zero     = np.zeros(len(actual_vals))
    naive_momentum = true_test.groupby("ticker")["returns_pct"].shift(1).fillna(0).values[:len(actual_vals)]

    # ── MAE comparison ────────────────────────────────────────────────────────
    mae_model    = mean_absolute_error(actual_vals, pred_median[:len(actual_vals)])
    mae_zero     = mean_absolute_error(actual_vals, naive_zero)
    mae_momentum = mean_absolute_error(actual_vals, naive_momentum)

    print("\n=== BASELINE COMPARISON ===")
    print(f"  MAE - TFT model      : {mae_model:.6f}")
    print(f"  MAE - Naive zero     : {mae_zero:.6f}")
    print(f"  MAE - Naive momentum : {mae_momentum:.6f}")

    if mae_model < mae_zero:
        improvement = (mae_zero - mae_model) / mae_zero * 100
        print(f"  ✓ Model beats naive zero by {improvement:.1f}%")
    else:
        print("  ✗ Model WORSE than predicting zero — no real signal")

    # ── Directional accuracy ──────────────────────────────────────────────────
    preds_aligned = pred_median[:len(actual_vals)]
    correct       = np.sign(preds_aligned) == np.sign(actual_vals)
    dir_acc       = correct.mean()

    print(f"\n  Directional accuracy : {dir_acc:.2%}")
    if dir_acc > 0.55:
        print("  ✓ Strong signal (>55%)")
    elif dir_acc > 0.52:
        print("  ~ Marginal signal (52-55%)")
    else:
        print("  ✗ No better than coin flip (<52%)")

    # ── Naive zero directional accuracy ──────────────────────────────────────
    zero_dir = (np.sign(naive_zero) == np.sign(actual_vals)).mean()
    print(f"  Naive zero dir acc   : {zero_dir:.2%}")
    print("=== END BASELINE COMPARISON ===\n")

if __name__ == "__main__":
    main()