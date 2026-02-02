import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

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
    print(f"[OK] Columns: {list(df.columns)}")
    
    unique_stocks = df['symbol'].unique()
    print(f"[OK] Found {len(unique_stocks)} unique stocks")
except Exception as e:
    print(f"[ERROR] Failed to load dataset: {e}")
    exit(1)

# Define temporal split
try:
    # Fill missing values
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
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

# Create datasets with actual column names
try:
    from pytorch_forecasting.data import TimeSeriesDataSet, TorchNormalizer
    
    print("[STATUS] Creating dataset objects...")
    
    target_normalizer = TorchNormalizer()
    
    # Use correct column names from actual parquet
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
            "open", "high", "low", "close", "volume",
            "returns_pct", "log_returns", "high_low_spread", "open_close_spread",
            "rolling_mean_7", "rolling_std_7", "rolling_std_30",
            "EMA_20", "EMA_50", "SMA_20", "RSI_14", "MACD", "MACD_signal", "MACD_hist",
            "ATR_14", "true_range", "bb_m", "bb_std", "bb_u", "bb_l", "bollinger_bandwidth",
            "volatility_20d", "volatility_60d",
            "sentiment_news_composite", "sentiment_social_score",
            "reddit_sentiment_mean", "reddit_post_volume", "reddit_sentiment_delta"
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
    print(f"[WARNING] Validation failed: {e}")

# Generate predictions on test set
print("\n" + "="*80)
print("TEST SET PREDICTIONS & EVALUATION")
print("="*80)

try:
    print("[STATUS] Generating predictions on test set...")
    predictions = best_model.predict(test_loader, mode="prediction", return_x=True)
    
    y_pred = predictions[0].cpu().numpy()
    x, y_actual = predictions[1]
    
    print(f"[OK] Predictions generated")
    print(f"  Predictions shape: {y_pred.shape}")
    print(f"  Actual targets shape: {y_actual.shape}")
    
except Exception as e:
    print(f"[ERROR] Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Calculate metrics
try:
    print("\n[STATUS] Computing metrics...")
    
    # Reshape if needed
    if len(y_pred.shape) == 3:
        y_pred = y_pred.squeeze(-1)
    
    if len(y_actual.shape) > 1:
        y_actual = y_actual.squeeze(-1) if y_actual.shape[-1] == 1 else y_actual
    
    # Denormalize using normalizer
    target_norm = training.target_normalizer
    if hasattr(target_norm, 'inverse_transform'):
        y_pred_denorm = target_norm.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
        y_actual_denorm = target_norm.inverse_transform(y_actual.reshape(-1, 1)).reshape(y_actual.shape)
    else:
        y_pred_denorm = y_pred
        y_actual_denorm = y_actual
    
    # Flatten and filter valid values
    y_pred_flat = y_pred_denorm.flatten()
    y_actual_flat = y_actual_denorm.flatten()
    
    valid_mask = ~(np.isnan(y_pred_flat) | np.isnan(y_actual_flat) | np.isinf(y_pred_flat) | np.isinf(y_actual_flat))
    y_pred_valid = y_pred_flat[valid_mask]
    y_actual_valid = y_actual_flat[valid_mask]
    
    print(f"[OK] Valid predictions: {len(y_pred_valid)} out of {len(y_pred_flat)}")
    
    # Calculate overall metrics
    mae = np.mean(np.abs(y_pred_valid - y_actual_valid))
    rmse = np.sqrt(np.mean((y_pred_valid - y_actual_valid) ** 2))
    mape = np.mean(np.abs((y_actual_valid - y_pred_valid) / (np.abs(y_actual_valid) + 1e-8))) * 100
    
    print(f"\n{'='*80}")
    print("[OK] OVERALL TEST SET METRICS:")
    print(f"{'='*80}")
    print(f"  MAE (Mean Absolute Error):        ${mae:.4f}")
    print(f"  RMSE (Root Mean Squared Error):   ${rmse:.4f}")
    print(f"  MAPE (Mean Absolute % Error):     {mape:.2f}%")
    print(f"  Number of valid predictions:      {len(y_pred_valid):,}")
    
    # Per-horizon metrics if 2D
    if len(y_pred.shape) == 2:
        print(f"\n{'='*80}")
        print("[OK] PER-HORIZON PREDICTION METRICS:")
        print(f"{'='*80}")
        print(f"  {'Horizon':<10} {'MAE':<15} {'RMSE':<15} {'MAPE %':<15}")
        print(f"  {'-'*55}")
        
        for h in range(min(y_pred.shape[1], 7)):
            pred_h = y_pred_denorm[:, h]
            actual_h = y_actual_denorm[:, h]
            
            valid_h = ~(np.isnan(pred_h) | np.isnan(actual_h) | np.isinf(pred_h) | np.isinf(actual_h))
            pred_valid_h = pred_h[valid_h]
            actual_valid_h = actual_h[valid_h]
            
            if len(pred_valid_h) > 0:
                mae_h = np.mean(np.abs(pred_valid_h - actual_valid_h))
                rmse_h = np.sqrt(np.mean((pred_valid_h - actual_valid_h) ** 2))
                mape_h = np.mean(np.abs((actual_valid_h - pred_valid_h) / (np.abs(actual_valid_h) + 1e-8))) * 100
                
                print(f"  {h+1} day(s)    ${mae_h:<14.4f} ${rmse_h:<14.4f} {mape_h:<14.2f}")
    
except Exception as e:
    print(f"[ERROR] Metrics calculation failed: {e}")
    import traceback
    traceback.print_exc()

# Model information
print(f"\n{'='*80}")
print("MODEL ARCHITECTURE & CONFIGURATION")
print(f"{'='*80}")

try:
    print(f"[OK] Model Type: Temporal Fusion Transformer (TFT)")
    print(f"  - Hidden Size: {best_model.hparams.hidden_size}")
    print(f"  - Attention Heads: {best_model.hparams.attention_head_size}")
    print(f"  - Dropout: {best_model.hparams.dropout}")
    print(f"  - Learning Rate: {best_model.hparams.learning_rate}")
    print(f"  - Encoder Length: 60 days (historical window)")
    print(f"  - Prediction Length: 7 days (forecast horizon)")
    
    print(f"\n[OK] Input Features ({len(training.time_varying_unknown_reals)} features):")
    print(f"  - Price Data: OHLCV")
    print(f"  - Returns: returns_pct, log_returns")
    print(f"  - Price Spreads: high_low_spread, open_close_spread")
    print(f"  - Moving Averages: rolling_mean_7, EMA_20, EMA_50, SMA_20")
    print(f"  - Volatility: rolling_std_7, rolling_std_30, volatility_20d, volatility_60d")
    print(f"  - Technical: RSI_14, MACD, MACD_signal, MACD_hist, ATR_14, true_range")
    print(f"  - Bollinger Bands: bb_m, bb_std, bb_u, bb_l, bollinger_bandwidth")
    print(f"  - Sentiment: sentiment_news_composite, sentiment_social_score")
    print(f"  - Reddit: reddit_sentiment_mean, reddit_post_volume, reddit_sentiment_delta")
    
    print(f"\n[OK] Training Configuration:")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Number of Stocks: {len(unique_stocks)}")
    print(f"  - Total Samples: {n_samples:,}")
    print(f"  - Train Samples: {len(train_data):,}")
    print(f"  - Validation Samples: {len(val_data):,}")
    print(f"  - Test Samples: {len(test_data):,}")
    
except Exception as e:
    print(f"[WARNING] Could not display model info: {e}")

print(f"\n{'='*80}")
print("EVALUATION COMPLETE")
print(f"{'='*80}\n")
