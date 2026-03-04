"""
Simple evaluation script for TFT no-sentiment model.
Uses direct data extraction without relying on complex batch structure.
"""

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from torch.serialization import safe_globals


print("=" * 70)
print("TFT NO-SENTIMENT MODEL - SIMPLE EVALUATION")
print("=" * 70)

# Load data
print("\n[1] Loading data...")
df = pd.read_parquet("data/processed/tft_full_universe.parquet")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["symbol", "time_idx"]).reset_index(drop=True)

test_df = df[df.date >= "2024-01-01"].copy()

TARGET_COLS = [f"target_future_{i}" for i in range(1, 8)]
test_df = test_df.dropna(subset=TARGET_COLS).reset_index(drop=True)

print(f"   Test set size: {len(test_df)} rows")
print(f"   Symbols: {test_df['symbol'].nunique()}")
print(f"   Date range: {test_df['date'].min()} to {test_df['date'].max()}")

# Load model
print("\n[2] Loading model...")
with safe_globals([MultiNormalizer, GroupNormalizer]):
    model = TemporalFusionTransformer.load_from_checkpoint(
        "models/tft/tft_leakage_safe_no_sentiment.ckpt",
        map_location="cpu",
        weights_only=False,
    )
model.eval()
print("   Model loaded")

# Create dataset
print("\n[3] Creating dataset...")
test_ds = TimeSeriesDataSet.from_parameters(
    model.dataset_parameters,
    test_df,
    predict=True,
    stop_randomization=True,
)

test_loader = test_ds.to_dataloader(train=False, batch_size=64, shuffle=False)

print(f"   Dataset: {len(test_ds)} sequences")
print(f"   Dataloader: {len(test_loader)} batches")

# Get predictions
print("\n[4] Generating predictions...")
preds_raw = model.predict(test_loader, mode="prediction", return_x=False)

# Concatenate predictions
preds = np.concatenate([np.asarray(p) for p in preds_raw], axis=0)
print(f"   Predictions shape: {preds.shape}")
print(f"   Predictions range: [{preds.min():.2f}, {preds.max():.2f}]")

# Extract ground truth from test_df
print("\n[5] Extracting ground truth...")
y_true = test_df[TARGET_COLS].values
symbols = test_df['symbol'].values

print(f"   Ground truth shape: {y_true.shape}")
print(f"   Symbols shape: {symbols.shape}")

# Make sure shapes are compatible
if preds.shape[0] != y_true.shape[0]:
    print(f"\n   WARNING: Shape mismatch (preds={preds.shape[0]}, targets={y_true.shape[0]})")
    print(f"   Using first {min(preds.shape[0], y_true.shape[0])} samples...")
    min_n = min(preds.shape[0], y_true.shape[0])
    preds = preds[:min_n]
    y_true = y_true[:min_n]
    symbols = symbols[:min_n]

# Calculate metrics
print("\n" + "=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

mae_list = []
rmse_list = []
mape_list = []

for sym in sorted(np.unique(symbols)):
    mask = symbols == sym
    sym_preds = preds[mask]
    sym_true = y_true[mask]
    
    mae = np.mean(np.abs(sym_preds - sym_true))
    rmse = np.sqrt(np.mean((sym_preds - sym_true) ** 2))
    mape = np.mean(np.abs((sym_true - sym_preds) / (np.abs(sym_true) + 1e-8))) * 100
    
    print(f"{sym:6s} | n={len(sym_preds):4d} MAE={mae:8.4f} RMSE={rmse:8.4f} MAPE={mape:6.2f}%")
    
    mae_list.append(mae)
    rmse_list.append(rmse)
    mape_list.append(mape)

print("\n" + "-" * 70)
print(f"MEAN   | MAE={np.mean(mae_list):8.4f} RMSE={np.mean(rmse_list):8.4f} MAPE={np.mean(mape_list):6.2f}%")
print(f"STD    | MAE={np.std(mae_list):8.4f} RMSE={np.std(rmse_list):8.4f} MAPE={np.std(mape_list):6.2f}%")
print("=" * 70)
