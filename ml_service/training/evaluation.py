import numpy as np
import pandas as pd

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from torch.serialization import safe_globals


MODEL_PATH = "models/tft/tft_leakage_safe_no_sentiment.ckpt"
DATA_PATH = "data/processed/tft_full_universe.parquet"

TARGET_COLS = [f"target_future_{i}" for i in range(1, 8)]


# ===============================
# LOAD DATA
# ===============================
df = pd.read_parquet(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

test_df = df[df.date >= "2024-01-01"].copy()
symbols = test_df["symbol"].unique()

print(f"Test rows: {len(test_df)}")
print(f"Symbols: {symbols}")


# ===============================
# LOAD MODEL SAFELY
# ===============================
with safe_globals([MultiNormalizer, GroupNormalizer]):
    model = TemporalFusionTransformer.load_from_checkpoint(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

model.eval()
print("✔ Model loaded successfully")


# ===============================
# METRIC STORAGE
# ===============================
mae_list = []
rmse_list = []
dir_acc_list = []
symbol_list = []


# ===============================
# PER-SYMBOL EVALUATION
# ===============================
for symbol in symbols:
    df_sym = test_df[test_df.symbol == symbol].copy()

    if len(df_sym) < 10:
        continue

    # Predictions → (num_samples, 7)
    preds = np.array(
        model.predict(df_sym, mode="prediction")
    )

    # Ground truth
    y_true = df_sym[TARGET_COLS].values

    # Safety check
    min_len = min(len(preds), len(y_true))
    preds = preds[:min_len]
    y_true = y_true[:min_len]

    # Metrics
    mae = np.mean(np.abs(preds - y_true))
    rmse = np.sqrt(np.mean((preds - y_true) ** 2))

    # Directional accuracy: does predicted return sign match actual return sign?
    dir_acc = np.mean(
        np.sign(preds[:, 0]) == np.sign(y_true[:, 0])
    )

    mae_list.append(mae)
    rmse_list.append(rmse)
    dir_acc_list.append(dir_acc)
    symbol_list.append(symbol)


# ===============================
# DISPLAY RESULTS IN TABLE FORMAT
# ===============================
results_df = pd.DataFrame({
    'Stock': symbol_list,
    'MAE': mae_list,
    'RMSE': rmse_list,
    'DIR_ACC': [f"{x:.1%}" for x in dir_acc_list]
})

print("\n" + "="*60)
print("EVALUATION RESULTS - ALL STOCKS")
print("="*60)
print(results_df.to_string(index=False))

print("\n" + "="*60)
print("FINAL AGGREGATED RESULTS")
print("="*60)
print(f"{'MAE':20s}: {np.mean(mae_list):.6f} ± {np.std(mae_list):.6f}")
print(f"{'RMSE':20s}: {np.mean(rmse_list):.6f} ± {np.std(rmse_list):.6f}")
print(f"{'DIR_ACC':20s}: {np.mean(dir_acc_list):.1%} ± {np.std(dir_acc_list):.1%}")
print("="*60)
