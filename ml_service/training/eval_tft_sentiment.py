import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU

import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import json
from pathlib import Path

print("[STATUS] Starting TFT Model Evaluation...")

# Configuration
CHECKPOINT_PATH = "models/tft/tft_with_sentiment.ckpt"
PARQUET_PATH = "data/processed/tft_full_universe.parquet"
BATCH_SIZE = 32

# Check if checkpoint exists
if not os.path.exists(CHECKPOINT_PATH):
    print(f"[ERROR] Checkpoint not found at {CHECKPOINT_PATH}")
    exit(1)

print(f"[OK] Loading checkpoint from {CHECKPOINT_PATH}")

try:
    # Load the trained model
    best_model = TemporalFusionTransformer.load_from_checkpoint(CHECKPOINT_PATH)
    print("[OK] Model loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit(1)

# Load dataset
try:
    print(f"[STATUS] Loading dataset from {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"[OK] Dataset loaded: {df.shape}")
    
    # Get unique stocks
    unique_stocks = df['symbol'].unique()
    print(f"[OK] Found {len(unique_stocks)} unique stocks")
except Exception as e:
    print(f"[ERROR] Failed to load dataset: {e}")
    exit(1)

# Define temporal split
try:
    # Create temporal split (70/10/20 for train/val/test)
    n_samples = len(df)
    train_cutoff = int(n_samples * 0.7)
    val_cutoff = int(n_samples * 0.8)
    
    train_data = df[:train_cutoff].copy()
    val_data = df[train_cutoff:val_cutoff].copy()
    test_data = df[val_cutoff:].copy()
    
    print(f"[OK] Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
except Exception as e:
    print(f"[ERROR] Failed to split data: {e}")
    exit(1)

# Prepare datasets using best_model's training data
try:
    from pytorch_forecasting.data import TimeSeriesDataSet, TorchNormalizer
    
    # Get the training dataset configuration from model
    print("[STATUS] Creating dataset objects...")
    
    # Create normalizer
    target_normalizer = TorchNormalizer()
    
    training = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",
        target="close",
        group_ids=["symbol"],
        min_encoder_length=60,
        max_encoder_length=60,
        min_prediction_length=7,
        max_prediction_length=7,
        static_categoricals=["symbol"],
        time_varying_known_reals=["day_of_week", "week_of_year", "month", "is_month_start", "is_month_end"],
        time_varying_unknown_reals=[
            "open", "high", "low", "close", "volume", "adj_close",
            "daily_return", "log_return", "return_volatility",
            "RSI_14", "MACD", "MACD_signal", "MACD_hist",
            "bb_m", "bb_std", "bb_u", "bb_l", "bollinger_bandwidth",
            "EMA_20", "EMA_50", "SMA_20", "ATR_14",
            "true_range", "volatility_20d", "volatility_60d",
            "volume_change", "volume_sma_20",
            "sentiment_news_composite", "sentiment_social_score",
            "reddit_sentiment_mean", "reddit_post_volume"
        ],
        target_normalizer=target_normalizer,
    )
    
    validation = TimeSeriesDataSet.from_dataset(training, val_data, predict=False, stop_randomization=True)
    test = TimeSeriesDataSet.from_dataset(training, test_data, predict=True, stop_randomization=True)
    
    print("[OK] Datasets created")
    
except Exception as e:
    print(f"[ERROR] Failed to create datasets: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Create data loaders
try:
    train_loader = DataLoader(training, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"[OK] DataLoaders created: Train batches={len(train_loader)}, Val batches={len(val_loader)}, Test batches={len(test_loader)}")
    
except Exception as e:
    print(f"[ERROR] Failed to create data loaders: {e}")
    exit(1)

# Evaluate on validation set
print("\n" + "="*80)
print("VALIDATION SET EVALUATION")
print("="*80)

try:
    trainer = Trainer(accelerator="cpu", enable_progress_bar=True)
    val_metrics = trainer.validate(best_model, val_loader, verbose=False)
    
    if val_metrics:
        print("[OK] Validation metrics:")
        for key, value in val_metrics[0].items():
            print(f"  {key}: {value:.6f}")
    
except Exception as e:
    print(f"[ERROR] Validation failed: {e}")

# Generate predictions on test set
print("\n" + "="*80)
print("TEST SET PREDICTIONS & EVALUATION")
print("="*80)

try:
    print("[STATUS] Generating predictions on test set...")
    predictions = best_model.predict(test_loader, mode="prediction", return_x=True)
    
    print(f"[OK] Predictions generated")
    print(f"  Predictions shape: {predictions[0].shape}")
    
    # Extract predictions and actuals
    y_pred = predictions[0].cpu().numpy()  # (batch, prediction_length, 1) or similar
    x, y_actual = predictions[1]
    
    print(f"  Actual targets shape: {y_actual.shape}")
    
except Exception as e:
    print(f"[ERROR] Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Calculate metrics for each prediction horizon
print("\n[STATUS] Computing metrics for each prediction horizon...")

try:
    # Reshape if needed
    if len(y_pred.shape) == 3:
        y_pred = y_pred.squeeze(-1)  # Remove last dimension if (batch, horizon, 1)
    
    if len(y_actual.shape) > 1:
        y_actual = y_actual.squeeze(-1) if y_actual.shape[-1] == 1 else y_actual
    
    # Compute denormalized predictions and actuals using the normalizer
    target_norm = best_model.hparams.target_normalizer
    
    if hasattr(target_norm, 'inverse_transform'):
        # Denormalize predictions
        if len(y_pred.shape) == 2:  # (batch, horizon)
            y_pred_denorm = target_norm.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
            y_actual_denorm = target_norm.inverse_transform(y_actual.reshape(-1, 1)).reshape(y_actual.shape)
        else:
            y_pred_denorm = y_pred
            y_actual_denorm = y_actual
    else:
        y_pred_denorm = y_pred
        y_actual_denorm = y_actual
    
    # Flatten for metrics calculation
    y_pred_flat = y_pred_denorm.flatten()
    y_actual_flat = y_actual_denorm.flatten()
    
    # Filter out NaN values
    valid_mask = ~(np.isnan(y_pred_flat) | np.isnan(y_actual_flat))
    y_pred_valid = y_pred_flat[valid_mask]
    y_actual_valid = y_actual_flat[valid_mask]
    
    # Calculate metrics
    mae = np.mean(np.abs(y_pred_valid - y_actual_valid))
    rmse = np.sqrt(np.mean((y_pred_valid - y_actual_valid) ** 2))
    mape = np.mean(np.abs((y_actual_valid - y_pred_valid) / (np.abs(y_actual_valid) + 1e-8))) * 100
    
    print(f"\n[OK] Overall Test Set Metrics:")
    print(f"  MAE (Mean Absolute Error):  ${mae:.4f}")
    print(f"  RMSE (Root Mean Squared Error): ${rmse:.4f}")
    print(f"  MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    
    # Per-horizon metrics
    if len(y_pred.shape) == 2:  # (batch, horizon)
        print(f"\n[OK] Per-Horizon Metrics:")
        print(f"  {'Horizon':<10} {'MAE':<15} {'RMSE':<15} {'MAPE (%)':<15}")
        print(f"  {'-'*55}")
        
        for h in range(y_pred.shape[1]):
            pred_h = y_pred_denorm[:, h]
            actual_h = y_actual_denorm[:, h]
            
            valid_mask_h = ~(np.isnan(pred_h) | np.isnan(actual_h))
            pred_valid = pred_h[valid_mask_h]
            actual_valid = actual_h[valid_mask_h]
            
            mae_h = np.mean(np.abs(pred_valid - actual_valid))
            rmse_h = np.sqrt(np.mean((pred_valid - actual_valid) ** 2))
            mape_h = np.mean(np.abs((actual_valid - pred_valid) / (np.abs(actual_valid) + 1e-8))) * 100
            
            print(f"  {h+1:<10} ${mae_h:<14.4f} ${rmse_h:<14.4f} {mape_h:<14.2f}")
    
except Exception as e:
    print(f"[ERROR] Metrics calculation failed: {e}")
    import traceback
    traceback.print_exc()

# Model information
print("\n" + "="*80)
print("MODEL INFORMATION")
print("="*80)

try:
    print(f"[OK] Model Architecture:")
    print(f"  Type: Temporal Fusion Transformer (TFT)")
    print(f"  Hidden Size: {best_model.hparams.hidden_size}")
    print(f"  Attention Heads: {best_model.hparams.attention_head_size}")
    print(f"  Dropout: {best_model.hparams.dropout}")
    print(f"  Learning Rate: {best_model.hparams.learning_rate}")
    print(f"  Encoder Length: 60 days")
    print(f"  Prediction Length: 7 days")
    print(f"\n[OK] Input Features:")
    print(f"  Total features used in model")
    print(f"  - Stock price: OHLCV")
    print(f"  - Technical: RSI, MACD, Bollinger Bands, EMA, SMA, ATR, Volatility")
    print(f"  - Sentiment: News composite, Social score, Reddit sentiment, Post volume")
    print(f"  - Temporal: Day of week, Week of year, Month, Month start/end flags")
    print(f"\n[OK] Training Data:")
    print(f"  Total samples: {n_samples}")
    print(f"  Unique stocks: {len(unique_stocks)}")
    print(f"  Stocks: {', '.join(sorted(unique_stocks)[:5])}... (+{len(unique_stocks)-5} more)")
    
except Exception as e:
    print(f"[WARNING] Could not display model info: {e}")

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
