"""
Rolling Window Evaluation for TFT Model
========================================

Generates 250+ predictions per stock by creating rolling TimeSeriesDataSet sequences
across the entire test period (2024).

For each position:
- Uses past 60 days as encoder input
- Predicts next 7-day returns
- Compares with actual returns
- Tracks directional accuracy by horizon (Day 1, 3, 5, 7)

No future leakage - encoder uses only past data.
"""

import numpy as np
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from torch.serialization import safe_globals
import warnings
warnings.filterwarnings('ignore')

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "models/tft/tft_leakage_safe_no_sentiment.ckpt"
DATA_PATH = "data/processed/tft_full_universe.parquet"
ENCODER_LENGTH = 60
PREDICTION_LENGTH = 7

# ===============================
# LOAD DATA & MODEL
# ===============================
print("Loading data...")
df = pd.read_parquet(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["symbol", "time_idx"]).reset_index(drop=True)

test_df = df[df.date >= "2024-01-01"].copy()
print(f"Test set: {len(test_df)} rows | {test_df['date'].min().date()} to {test_df['date'].max().date()}")

print("\nLoading model...")
with safe_globals([MultiNormalizer, GroupNormalizer]):
    model = TemporalFusionTransformer.load_from_checkpoint(
        MODEL_PATH, map_location="cpu", weights_only=False
    )
model.eval()
print("✔ Model loaded")

# ===============================
# ROLLING WINDOW EVALUATION
# ===============================
TARGET_COLS = [f"target_future_{i}" for i in range(1, PREDICTION_LENGTH + 1)]
results_per_stock = []

print(f"\nRunning rolling window evaluation...")
print(f"One TimeSeriesDataSet per stock with automatic rolling windows...\n")

for symbol in sorted(test_df["symbol"].unique()):
    sym_data = test_df[test_df.symbol == symbol].copy().reset_index(drop=True)
    
    # Skip if not enough data
    if len(sym_data) < ENCODER_LENGTH + PREDICTION_LENGTH + 1:
        print(f"{symbol:6s} | SKIP - {len(sym_data)} rows")
        continue
    
    try:
        # Create TimeSeriesDataSet for entire stock
        # This automatically creates rolling windows
        stock_ds = TimeSeriesDataSet(
            sym_data,
            time_idx="time_idx",
            target=TARGET_COLS,
            group_ids=["symbol"],
            static_categoricals=["symbol"],
            time_varying_known_reals=[
                "time_idx", "day_of_week", "week_of_year", "month",
                "is_month_start", "is_month_end"
            ],
            time_varying_unknown_reals=[
                "open", "high", "low", "close", "volume",
                "high_low_spread", "open_close_spread",
                "rolling_mean_7", "rolling_std_7", "rolling_std_30",
                "EMA_20", "EMA_50", "SMA_20", "RSI_14",
                "MACD", "MACD_signal", "MACD_hist",
                "ATR_14", "true_range", "bollinger_bandwidth",
                "volatility_20d", "volatility_60d"
            ],
            max_encoder_length=ENCODER_LENGTH,
            max_prediction_length=PREDICTION_LENGTH,
            target_normalizer=MultiNormalizer(
                [GroupNormalizer(groups=["symbol"]) for _ in range(PREDICTION_LENGTH)]
            ),
            allow_missing_timesteps=False
        )
        
        # Create dataloader and predict once
        loader = stock_ds.to_dataloader(train=False, batch_size=256, shuffle=False)
        all_preds = model.predict(loader, mode="prediction", return_x=False)
        pred_array = np.concatenate([np.asarray(p) for p in all_preds], axis=0)
        
        # Get actuals: TimeSeriesDataSet creates sequences starting at encoder_length
        actual_array = sym_data.iloc[ENCODER_LENGTH:ENCODER_LENGTH + len(pred_array)][TARGET_COLS].values
        
        # Match lengths
        min_len = min(len(pred_array), len(actual_array))
        pred_array = pred_array[:min_len]
        actual_array = actual_array[:min_len]
        
        if min_len > 0:
            # Compute metrics
            mae = np.mean(np.abs(pred_array - actual_array))
            rmse = np.sqrt(np.mean((pred_array - actual_array) ** 2))
            
            # Directional accuracy (indices: 0=day1, 2=day3, 4=day5, 6=day7)
            dir_acc_d1 = 100.0 * np.mean(np.sign(pred_array[:, 0]) == np.sign(actual_array[:, 0]))
            dir_acc_d3 = 100.0 * np.mean(np.sign(pred_array[:, 2]) == np.sign(actual_array[:, 2]))
            dir_acc_d5 = 100.0 * np.mean(np.sign(pred_array[:, 4]) == np.sign(actual_array[:, 4]))
            dir_acc_d7 = 100.0 * np.mean(np.sign(pred_array[:, 6]) == np.sign(actual_array[:, 6]))
            
            results_per_stock.append({
                'Stock': symbol,
                'MAE': mae,
                'RMSE': rmse,
                'DIR_ACC_D1': dir_acc_d1,
                'DIR_ACC_D3': dir_acc_d3,
                'DIR_ACC_D5': dir_acc_d5,
                'DIR_ACC_D7': dir_acc_d7,
                'N_Predictions': min_len
            })
            
            print(f"{symbol:6s} | n={min_len:4d} | MAE={mae:.5f} | DIR_ACC(D7)={dir_acc_d7:5.1f}%")
        else:
            print(f"{symbol:6s} | FAIL - shape mismatch")
    
    except Exception as e:
        err = str(e)[:80]
        print(f"{symbol:6s} | ERROR - {err}")

# ===============================
# DISPLAY RESULTS
# ===============================
if results_per_stock:
    results_df = pd.DataFrame(results_per_stock)
    
    print("\n" + "="*130)
    print("PER-STOCK ROLLING EVALUATION RESULTS")
    print("="*130)
    print(results_df.to_string(index=False))
    
    print("\n" + "="*130)
    print("FINAL AGGREGATED METRICS")
    print("="*130)
    print(f"{'MAE':40s}: {results_df['MAE'].mean():.6f} ± {results_df['MAE'].std():.6f}")
    print(f"{'RMSE':40s}: {results_df['RMSE'].mean():.6f} ± {results_df['RMSE'].std():.6f}")
    print(f"{'Directional Accuracy (Day 1)':40s}: {results_df['DIR_ACC_D1'].mean():6.1f}% ± {results_df['DIR_ACC_D1'].std():5.1f}%")
    print(f"{'Directional Accuracy (Day 3)':40s}: {results_df['DIR_ACC_D3'].mean():6.1f}% ± {results_df['DIR_ACC_D3'].std():5.1f}%")
    print(f"{'Directional Accuracy (Day 5)':40s}: {results_df['DIR_ACC_D5'].mean():6.1f}% ± {results_df['DIR_ACC_D5'].std():5.1f}%")
    print(f"{'Directional Accuracy (Day 7)':40s}: {results_df['DIR_ACC_D7'].mean():6.1f}% ± {results_df['DIR_ACC_D7'].std():5.1f}%")
    print(f"{'Total Predictions Generated':40s}: {int(results_df['N_Predictions'].sum()):,}")
    print(f"{'Average Predictions per Stock':40s}: {int(results_df['N_Predictions'].mean())}")
    print("="*130)
else:
    print("\n❌ No results generated!")
