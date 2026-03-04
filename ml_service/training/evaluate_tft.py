"""
Evaluation script for TFT model trained WITHOUT sentiment columns.

Key insight: TimeSeriesDataSet creates sequences grouped by symbol.
With 37 unique stocks and training up to 2024, we get 37 sequences for evaluation.
Each sequence produces 1 prediction for the 7-day horizon.
"""

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from torch.serialization import safe_globals


print("=" * 70)
print("TFT NO-SENTIMENT MODEL EVALUATION")
print("=" * 70)

# Load data
print("\n[1] Loading data...")
df = pd.read_parquet("data/processed/tft_full_universe.parquet")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["symbol", "time_idx"]).reset_index(drop=True)

test_df = df[df.date >= "2024-01-01"].copy()
print(f"   Test rows (raw): {len(test_df)}")

TARGET_COLS = [f"target_future_{i}" for i in range(1, 8)]
test_df = test_df.dropna(subset=TARGET_COLS).reset_index(drop=True)
print(f"   Test rows (cleaned): {len(test_df)}")
print(f"   Unique symbols: {len(test_df['symbol'].unique())}")

# Load model
print("\n[2] Loading model...")
with safe_globals([MultiNormalizer, GroupNormalizer]):
    model = TemporalFusionTransformer.load_from_checkpoint(
        "models/tft/tft_leakage_safe_no_sentiment.ckpt",
        map_location="cpu",
        weights_only=False,
    )
model.eval()
print("   Model loaded successfully")

# Create dataset
print("\n[3] Creating TimeSeriesDataSet...")
test_ds = TimeSeriesDataSet.from_parameters(
    model.dataset_parameters,
    test_df,
    predict=True,
    stop_randomization=True,
)
print(f"   Dataset length: {len(test_ds)} sequences")

test_loader = test_ds.to_dataloader(
    train=False,
    batch_size=64,
    shuffle=False,
)
print(f"   Dataloader batches: {len(test_loader)}")

# Generate predictions
print("\n[4] Generating predictions...")
predictions_list = model.predict(test_loader, mode="prediction", return_x=False)
preds = np.concatenate([np.asarray(p) for p in predictions_list], axis=0)
print(f"   Predictions shape: {preds.shape}")
print(f"   (num_sequences=37, prediction_horizon=7)")

# Extract targets and symbols  
print("\n[5] Extracting targets and symbols...")
all_targets = []
all_symbols = []
symbol_map = {i: sym for i, sym in enumerate(sorted(test_df['symbol'].unique()))}

for batch_idx, (x_dict, _) in enumerate(test_loader):
    # decoder_target comes as a list from the TimeSeriesDataSet
    targets_list = x_dict['decoder_target']
    groups_list = x_dict['groups']
    
    # Convert to tensors/arrays if needed
    if isinstance(targets_list, list):
        targets_array = torch.stack([torch.tensor(t) if not isinstance(t, torch.Tensor) else t 
                                     for t in targets_list]).numpy()
    else:
        targets_array = targets_list.numpy() if isinstance(targets_list, torch.Tensor) else targets_list
    
    if isinstance(groups_list, list):
        groups_array = torch.stack([torch.tensor(g) if not isinstance(g, torch.Tensor) else g 
                                    for g in groups_list]).numpy()
    else:
        groups_array = groups_list.numpy() if isinstance(groups_list, torch.Tensor) else groups_list
    
    # targets_array shape: (seq_len, num_targets) where seq_len is 60
    # We need the last timestep (the actual 7-day targets)
    last_targets = targets_array[-1, :]  # Shape: (7,)
    all_targets.append(last_targets)
    
    # groups_array shape: (seq_len, num_vars) where vars include symbol idx
    # Symbol is in first column, consistent across sequence
    sym_idx = int(groups_array[0, 0])
    sym_name = symbol_map.get(sym_idx, f"unknown_{sym_idx}")
    all_symbols.append(sym_name)
    
    if batch_idx >= 5:
        print(f"   Processed batch {batch_idx}, found targets shape: {last_targets.shape}, symbol: {sym_name}")

y_true = np.array(all_targets)
symbols = np.array(all_symbols)

print(f"\n   Final data shapes:")
print(f"   - Predictions: {preds.shape}")
print(f"   - Targets: {y_true.shape}")
print(f"   - Symbols: {symbols.shape}")

if preds.shape[0] == y_true.shape[0] == len(symbols):
    print("   [OK] All shapes match!")
else:
    print(f"   [WARNING] Shape mismatch!")
    min_len = min(preds.shape[0], y_true.shape[0], len(symbols))
    preds = preds[:min_len]
    y_true = y_true[:min_len]
    symbols = symbols[:min_len]
    print(f"   Truncated to {min_len} sequences")

# Calculate metrics
print("\n" + "=" * 70)
print("PER-SYMBOL EVALUATION RESULTS")
print("=" * 70)

mae_list = []
rmse_list = []
mape_list = []
dir_acc_list = []

for sym in sorted(np.unique(symbols)):
    mask = symbols == sym
    sym_preds = preds[mask]
    sym_true = y_true[mask]
    
    if len(sym_preds) == 0:
        continue
    
    # Metrics
    mae = np.mean(np.abs(sym_preds - sym_true))
    rmse = np.sqrt(np.mean((sym_preds - sym_true) ** 2))
    mape = np.mean(np.abs((sym_true - sym_preds) / (np.abs(sym_true) + 1e-8))) * 100
    
    # Directional accuracy: does predicted return sign match actual return sign?
    dir_acc = np.mean(np.sign(sym_preds) == np.sign(sym_true)) * 100
    
    print(f"{sym:6s} | MAE={mae:8.4f} RMSE={rmse:8.4f} MAPE={mape:6.2f}% DIR_ACC={dir_acc:5.1f}% (n={len(sym_preds)})")
    
    mae_list.append(mae)
    rmse_list.append(rmse)
    mape_list.append(mape)
    dir_acc_list.append(dir_acc)

# Aggregate metrics
print("\n" + "=" * 70)
print("AGGREGATE METRICS")
print("=" * 70)
print(f"MAE              : {np.mean(mae_list):8.4f} +/- {np.std(mae_list):6.4f}")
print(f"RMSE             : {np.mean(rmse_list):8.4f} +/- {np.std(rmse_list):6.4f}")
print(f"MAPE             : {np.mean(mape_list):8.2f}% +/- {np.std(mape_list):5.2f}%")
print(f"Directional Acc  : {np.mean(dir_acc_list):8.1f}% +/- {np.std(dir_acc_list):5.1f}%")
print(f"Num sequences    : {len(mae_list)}")
print("=" * 70)
