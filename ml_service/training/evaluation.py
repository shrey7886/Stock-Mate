import numpy as np
import pandas as pd

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from torch.serialization import safe_globals


MODEL_PATH = "models/tft/tft_no_sentiment.ckpt"
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

    dir_acc = np.mean(
        np.sign(preds[:, 0] - df_sym["close"].values[:min_len])
        == np.sign(y_true[:, 0] - df_sym["close"].values[:min_len])
    )

    print(f"{symbol} | MAE={mae:.4f} RMSE={rmse:.4f} DIR_ACC={dir_acc:.2%}")

    mae_list.append(mae)
    rmse_list.append(rmse)
    dir_acc_list.append(dir_acc)


# ===============================
# FINAL AGGREGATE METRICS
# ===============================
print("\n📊 FINAL AGGREGATED RESULTS")
print(f"MAE  : {np.mean(mae_list):.4f}")
print(f"RMSE : {np.mean(rmse_list):.4f}")
print(f"Directional Accuracy (t+1): {np.mean(dir_acc_list):.2%}")
