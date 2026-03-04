import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from torch.serialization import safe_globals


# ===============================
# CONFIG
# ===============================
MODEL_PATH = "models/tft/tft_no_sentiment.ckpt"
DATA_PATH = "data/processed/tft_full_universe.parquet"

BATCH_SIZE = 64
TARGET_COLS = [f"target_future_{i}" for i in range(1, 8)]


# ===============================
# LOAD DATA
# ===============================
print("Loading data...")
df = pd.read_parquet(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["symbol", "time_idx"]).reset_index(drop=True)

# Use data from 2024 onwards for testing
test_df = df[df.date >= "2024-01-01"].copy()

print(f"Test rows: {len(test_df)}")
print(f"Symbols: {sorted(test_df.symbol.unique())}")
print(f"Columns in data: {test_df.columns.tolist()}")

# Drop rows with NaN targets
test_df = test_df.dropna(subset=TARGET_COLS).reset_index(drop=True)
print(f"Test rows after cleanup: {len(test_df)}\n")


# ===============================
# LOAD MODEL
# ===============================
print("Loading model...")
with safe_globals([MultiNormalizer, GroupNormalizer]):
    model = TemporalFusionTransformer.load_from_checkpoint(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

model.eval()
print("Model loaded successfully\n")

# Check model's dataset parameters
print(f"Model dataset parameters keys: {model.dataset_parameters.keys()}")


# ===============================
# REBUILD DATASET FROM MODEL
# ===============================
print("\nRebuilding dataset...")
test_ds = TimeSeriesDataSet.from_parameters(
    model.dataset_parameters,
    test_df,
    predict=True,
    stop_randomization=True,
)

test_loader = test_ds.to_dataloader(
    train=False,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print(f"Test dataloader created with {len(test_loader)} batches")


# ===============================
# MAKE PREDICTIONS
# ===============================
print("\nGenerating predictions...")
predictions_list = model.predict(test_loader, mode="prediction", return_x=False)

# Convert predictions to numpy array
preds = np.concatenate([np.asarray(p) for p in predictions_list], axis=0)
print(f"Predictions shape: {preds.shape}")  # Should be (num_samples, 7)


# ===============================
# EXTRACT TRUE VALUES & SYMBOLS (Fixed)
# ===============================
print("Extracting ground truth and symbols...")
y_true_list = []
symbols_list = []
batch_counter = 0

# Iterate through the dataloader again - predictions and data must align
for batch in test_loader:
    x_dict, _ = batch
    batch_counter += 1
    
    # Get targets - shape: (batch_size, seq_len, num_targets)
    targets = x_dict["decoder_target"]
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    else:
        targets = np.array(targets)
    
    # Get batch size (number of sequences in this batch)
    batch_size = targets.shape[0]
    
    # For each sequence in this batch, extract last timestep targets
    for i in range(batch_size):
        y_true_list.append(targets[i, -1, :])  # Shape: (7,)
    
    # Get symbols from groups - shape: (seq_len, num_vars)
    groups = x_dict["groups"]
    if isinstance(groups, torch.Tensor):
        groups = groups.cpu().numpy()
    
    # groups shape: (seq_len, num_vars), we need one symbol per batch sequence
    # The symbol index is the same for all timesteps in a sequence
    seq_len = groups.shape[0]
    for i in range(batch_size):
        # Symbol is stored as an index in the groups tensor
        # Get symbol index - it's typically groups[0, 0] but let's check
        sym_idx = int(groups[0, 0]) if groups.ndim > 1 else int(groups[0])
        unique_symbols = sorted(test_df['symbol'].unique())
        sym_name = unique_symbols[sym_idx] if sym_idx < len(unique_symbols) else f"unknown_{sym_idx}"
        symbols_list.append(sym_name)

y_true = np.array(y_true_list)
symbols = np.array(symbols_list)

print(f"Ground truth shape: {y_true.shape}")
print(f"Symbols shape: {symbols.shape}")
print(f"Predictions shape: {preds.shape}")
print(f"\nShape verification:")
print(f"  preds:   {preds.shape[0]} samples")
print(f"  y_true:  {y_true.shape[0]} samples")
print(f"  symbols: {len(symbols)} labels")
print(f"  Match: {preds.shape[0] == y_true.shape[0] == len(symbols)}")

if preds.shape[0] != y_true.shape[0] or preds.shape[0] != len(symbols):
    min_len = min(preds.shape[0], y_true.shape[0], len(symbols))
    print(f"\nWARNING: Truncating to {min_len} samples to match shapes")
    preds = preds[:min_len]
    y_true = y_true[:min_len]
    symbols = symbols[:min_len]


# ===============================
# CALCULATE METRICS
# ===============================
print("\n" + "="*60)
print("PER-SYMBOL RESULTS")
print("="*60)

mae_list = []
rmse_list = []
dir_acc_list = []
symbol_counts = []

for sym in sorted(np.unique(symbols)):
    mask = symbols == sym
    sym_preds = preds[mask]
    sym_true = y_true[mask]
    
    if len(sym_preds) == 0:
        continue
    
    # Calculate metrics
    mae = np.mean(np.abs(sym_preds - sym_true))
    rmse = np.sqrt(np.mean((sym_preds - sym_true) ** 2))
    
    # Directional accuracy for t+1 (first prediction step)
    pred_direction = np.sign(sym_preds[:, 0])
    actual_direction = np.sign(sym_true[:, 0])
    dir_acc = np.mean(pred_direction == actual_direction)
    
    print(f"{sym:6s} | MAE={mae:8.4f} RMSE={rmse:8.4f} DIR_ACC={dir_acc:6.1%} (n={len(sym_preds)})")
    
    mae_list.append(mae)
    rmse_list.append(rmse)
    dir_acc_list.append(dir_acc)
    symbol_counts.append(len(sym_preds))


# ===============================
# AGGREGATE RESULTS
# ===============================
print("\n" + "="*60)
print("AGGREGATE RESULTS")
print("="*60)
print(f"MAE              : {np.mean(mae_list):.4f} (+/- {np.std(mae_list):.4f})")
print(f"RMSE             : {np.mean(rmse_list):.4f} (+/- {np.std(rmse_list):.4f})")
print(f"Directional Acc  : {np.mean(dir_acc_list):.2%} (+/- {np.std(dir_acc_list):.2%})")
print(f"Total samples    : {len(preds)}")
print(f"Total symbols    : {len(mae_list)}")
