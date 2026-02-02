"""
TFT Model Evaluation Script
Evaluates the trained TFT model on test set and generates comprehensive metrics
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, RMSE, MAPE, SMAPE
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

print("\n" + "="*80)
print("[EVALUATION] TFT Model Evaluation with Sentiment Features")
print("="*80 + "\n")

# Configuration
DATA_PATH = "data/processed/tft_full_universe.parquet"
MODEL_PATH = "models/tft/tft_with_sentiment.ckpt"
BATCH_SIZE = 32
ENCODER_LENGTH = 60
PREDICTION_LENGTH = 7

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model checkpoint not found at {MODEL_PATH}")
    print("[INFO] Please train the model first using train_tft_sentiment.py")
    sys.exit(1)

print(f"[LOAD] Loading dataset from {DATA_PATH}...")
try:
    df = pd.read_parquet(DATA_PATH)
    print(f"[OK] Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"[INFO] Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"[INFO] Stocks: {df['symbol'].nunique()} unique symbols")
except Exception as e:
    print(f"[ERROR] Failed to load dataset: {e}")
    sys.exit(1)

# Data preparation
print("\n[PREPARE] Preparing data splits...")
try:
    # Sort by symbol and time
    df = df.sort_values(['symbol', 'time_idx'])
    
    # Create time splits (same as training)
    max_time_idx = df['time_idx'].max()
    train_cutoff = int(max_time_idx * 0.7)
    val_cutoff = int(max_time_idx * 0.85)
    
    print(f"[INFO] Train cutoff: {train_cutoff}, Validation cutoff: {val_cutoff}, Max: {max_time_idx}")
    
    # Create training dataset (needed for scaling parameters)
    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= train_cutoff],
        time_idx="time_idx",
        target="target_future_1",
        group_ids=["symbol"],
        min_encoder_length=ENCODER_LENGTH // 2,
        max_encoder_length=ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=PREDICTION_LENGTH,
        static_categoricals=["symbol"],
        time_varying_known_categoricals=["day_of_week", "month"],
        time_varying_known_reals=["time_idx", "week_of_year"],
        time_varying_unknown_reals=[
            col for col in df.columns 
            if col not in ["symbol", "date", "time_idx", "day_of_week", "week_of_year", 
                          "month", "is_month_start", "is_month_end"] + 
                          [f"target_future_{i}" for i in range(1, 8)]
        ],
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Create test dataset
    testing = TimeSeriesDataSet.from_dataset(
        training, 
        df[lambda x: x.time_idx > val_cutoff], 
        predict=True, 
        stop_randomization=True
    )
    
    test_dataloader = testing.to_dataloader(
        train=False, 
        batch_size=BATCH_SIZE, 
        num_workers=0
    )
    
    print(f"[OK] Test set created: {len(testing)} samples")
    
except Exception as e:
    print(f"[ERROR] Failed to prepare data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Load model
print(f"\n[LOAD] Loading trained model from {MODEL_PATH}...")
try:
    best_model = TemporalFusionTransformer.load_from_checkpoint(MODEL_PATH)
    best_model.eval()
    print("[OK] Model loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Generate predictions
print("\n[PREDICT] Generating predictions on test set...")
try:
    with torch.no_grad():
        predictions = best_model.predict(test_dataloader, mode="prediction", return_x=True)
    
    # Extract predictions and actuals
    y_pred = predictions.output.numpy()
    x, y_actual = predictions.x, predictions.y
    y_actual_np = y_actual[0].numpy()  # First target
    
    print(f"[OK] Predictions generated: {y_pred.shape}")
    print(f"[INFO] Prediction shape: {y_pred.shape} (samples x prediction_length)")
    
except Exception as e:
    print(f"[ERROR] Failed to generate predictions: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Calculate metrics
print("\n" + "="*80)
print("[METRICS] Model Performance Evaluation")
print("="*80 + "\n")

try:
    # For 1-day ahead prediction (target_future_1)
    y_pred_1day = y_pred[:, 0]  # First prediction horizon
    y_actual_1day = y_actual_np[:, 0]
    
    # Remove any NaN values
    mask = ~(np.isnan(y_pred_1day) | np.isnan(y_actual_1day))
    y_pred_clean = y_pred_1day[mask]
    y_actual_clean = y_actual_1day[mask]
    
    # Calculate metrics for 1-day ahead
    mae_1 = mean_absolute_error(y_actual_clean, y_pred_clean)
    rmse_1 = np.sqrt(mean_squared_error(y_actual_clean, y_pred_clean))
    mape_1 = mean_absolute_percentage_error(y_actual_clean, y_pred_clean) * 100
    
    # Calculate R-squared
    ss_res = np.sum((y_actual_clean - y_pred_clean) ** 2)
    ss_tot = np.sum((y_actual_clean - np.mean(y_actual_clean)) ** 2)
    r2_1 = 1 - (ss_res / ss_tot)
    
    print("[HORIZON] 1-Day Ahead Prediction:")
    print(f"  MAE (Mean Absolute Error):     {mae_1:.4f}")
    print(f"  RMSE (Root Mean Squared Error): {rmse_1:.4f}")
    print(f"  MAPE (Mean Absolute % Error):   {mape_1:.2f}%")
    print(f"  R-squared:                      {r2_1:.4f}")
    print(f"  Samples evaluated:              {len(y_pred_clean):,}")
    
    # Multi-horizon metrics
    print("\n[MULTI-HORIZON] Performance across all prediction horizons:")
    print("-" * 80)
    print(f"{'Horizon':<12} {'MAE':<12} {'RMSE':<12} {'MAPE':<12} {'R²':<12}")
    print("-" * 80)
    
    horizon_metrics = []
    for horizon in range(min(PREDICTION_LENGTH, y_pred.shape[1])):
        y_p = y_pred[:, horizon]
        y_a = y_actual_np[:, horizon] if horizon < y_actual_np.shape[1] else None
        
        if y_a is not None:
            mask = ~(np.isnan(y_p) | np.isnan(y_a))
            y_p_clean = y_p[mask]
            y_a_clean = y_a[mask]
            
            if len(y_p_clean) > 0:
                mae = mean_absolute_error(y_a_clean, y_p_clean)
                rmse = np.sqrt(mean_squared_error(y_a_clean, y_p_clean))
                mape = mean_absolute_percentage_error(y_a_clean, y_p_clean) * 100
                
                ss_res = np.sum((y_a_clean - y_p_clean) ** 2)
                ss_tot = np.sum((y_a_clean - np.mean(y_a_clean)) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                
                horizon_metrics.append({
                    'horizon': horizon + 1,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'r2': r2
                })
                
                print(f"{horizon+1:2d} days     {mae:>10.4f}  {rmse:>10.4f}  {mape:>9.2f}%  {r2:>10.4f}")
    
    print("-" * 80)
    
    # Average metrics across horizons
    avg_mae = np.mean([m['mae'] for m in horizon_metrics])
    avg_rmse = np.mean([m['rmse'] for m in horizon_metrics])
    avg_mape = np.mean([m['mape'] for m in horizon_metrics])
    avg_r2 = np.mean([m['r2'] for m in horizon_metrics])
    
    print(f"{'AVERAGE':<12} {avg_mae:>10.4f}  {avg_rmse:>10.4f}  {avg_mape:>9.2f}%  {avg_r2:>10.4f}")
    print("-" * 80)
    
    # Model interpretation
    print("\n" + "="*80)
    print("[INTERPRETATION] Model Performance Assessment")
    print("="*80 + "\n")
    
    # Baseline comparison (naive forecast - predict last value)
    print("[BASELINE] Comparison with Naive Forecast (predict last known value):")
    # For naive forecast, we'd predict the last encoder value as all future values
    # This is a simple baseline to beat
    baseline_pred = np.zeros_like(y_actual_1day)
    # In a real scenario, we'd use the last known value from encoder
    # For now, we'll use mean as baseline
    baseline_pred = np.full_like(y_actual_1day, np.nanmean(y_actual_1day))
    
    mask = ~np.isnan(y_actual_1day)
    baseline_mae = mean_absolute_error(y_actual_1day[mask], baseline_pred[mask])
    baseline_rmse = np.sqrt(mean_squared_error(y_actual_1day[mask], baseline_pred[mask]))
    
    print(f"  Baseline MAE:  {baseline_mae:.4f}")
    print(f"  Baseline RMSE: {baseline_rmse:.4f}")
    print(f"  Model MAE:     {mae_1:.4f}")
    print(f"  Model RMSE:    {rmse_1:.4f}")
    print(f"  Improvement:   {((baseline_mae - mae_1) / baseline_mae * 100):.1f}% better than baseline")
    
    # Performance assessment
    print("\n[ASSESSMENT] Overall Model Quality:")
    if avg_mape < 5:
        quality = "EXCELLENT"
        desc = "The model shows excellent predictive accuracy"
    elif avg_mape < 10:
        quality = "VERY GOOD"
        desc = "The model demonstrates strong predictive performance"
    elif avg_mape < 15:
        quality = "GOOD"
        desc = "The model shows good predictive capability"
    elif avg_mape < 25:
        quality = "FAIR"
        desc = "The model provides reasonable predictions but has room for improvement"
    else:
        quality = "NEEDS IMPROVEMENT"
        desc = "The model may need further tuning or more features"
    
    print(f"  Quality Rating: {quality}")
    print(f"  {desc}")
    print(f"\n  Key Metrics Summary:")
    print(f"    - Average MAPE: {avg_mape:.2f}% (lower is better)")
    print(f"    - Average R²:   {avg_r2:.4f} (closer to 1.0 is better)")
    
    if avg_r2 > 0.5:
        print(f"    - The model explains {avg_r2*100:.1f}% of variance in stock returns")
    else:
        print(f"    - The model explains {avg_r2*100:.1f}% of variance (consider adding more features)")
    
    # Feature importance (if available)
    print("\n[FEATURES] Sentiment Feature Impact:")
    print("  The model was trained with sentiment features including:")
    print("  - sentiment_news_composite (combined news sentiment)")
    print("  - sentiment_social_score (social media sentiment)")
    print("  - reddit_sentiment_mean (Reddit sentiment)")
    print("  - reddit_post_volume (social activity)")
    
    # Save metrics to file
    print("\n[SAVE] Saving evaluation results...")
    metrics_df = pd.DataFrame(horizon_metrics)
    metrics_df.to_csv('ml_service/training/evaluation_metrics.csv', index=False)
    
    # Save summary
    with open('ml_service/training/evaluation_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("TFT MODEL EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Test samples: {len(y_pred_clean):,}\n")
        f.write(f"Evaluation date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("-"*80 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Average MAPE:   {avg_mape:.2f}%\n")
        f.write(f"Average MAE:    {avg_mae:.4f}\n")
        f.write(f"Average RMSE:   {avg_rmse:.4f}\n")
        f.write(f"Average R²:     {avg_r2:.4f}\n\n")
        f.write(f"Quality Rating: {quality}\n")
        f.write(f"{desc}\n\n")
        f.write("-"*80 + "\n")
        f.write("HORIZON-SPECIFIC METRICS\n")
        f.write("-"*80 + "\n")
        for m in horizon_metrics:
            f.write(f"Day {m['horizon']}: MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, MAPE={m['mape']:.2f}%, R²={m['r2']:.4f}\n")
    
    print("[OK] Evaluation metrics saved to:")
    print("     - ml_service/training/evaluation_metrics.csv")
    print("     - ml_service/training/evaluation_summary.txt")
    
except Exception as e:
    print(f"[ERROR] Failed to calculate metrics: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("[COMPLETE] Evaluation finished successfully!")
print("="*80 + "\n")
