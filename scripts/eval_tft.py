# scripts/eval_tft.py
"""
Evaluate the trained TFT model on validation data.
Loads the best checkpoint and computes RMSE, MAE, MAPE metrics.

Usage:
    python scripts/eval_tft.py
"""

from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import torch

# Add project root to path so we can import ml_service modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml_service.training.train_tft import load_processed_universe, prepare_datasets
from pytorch_forecasting import TemporalFusionTransformer

def find_best_checkpoint(ckpt_dir: Path):
    """Find the most recent checkpoint (best one saved by ModelCheckpoint)."""
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=os.path.getmtime)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return str(ckpts[-1])

def main():
    # Paths
    CKPT_DIR = ROOT / "ml_service" / "models" / "saved_models" / "tft"
    ckpt_path = find_best_checkpoint(CKPT_DIR)
    print(f"Loading checkpoint: {ckpt_path}\n")

    # Load data and prepare dataset (same as training)
    print("Loading processed data...")
    df = load_processed_universe()
    print(f"Loaded {len(df)} rows from {df['symbol'].nunique()} symbols")

    print("Preparing dataset...")
    dataset = prepare_datasets(df)
    print(f"Dataset size (windows): {len(dataset)}\n")

    # Create validation dataloader
    val_dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
    
    # Load model
    print(f"Loading model from checkpoint...")
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=False)
    model.eval()
    print(f"Model loaded. Total params: {sum(p.numel() for p in model.parameters()):,}\n")

    # Direct forward pass on batches
    print("Generating predictions on validation set...")
    all_predictions = []
    all_targets = []
    batch_count = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            batch_count += 1
            
            # Unpack batch: pytorch-forecasting returns (x_dict, y_tuple)
            if isinstance(batch, (tuple, list)):
                x, y = batch
            else:
                x = batch
                y = None
            
            # Forward pass
            try:
                output = model(x)
            except Exception as e:
                print(f"Error in forward pass: {e}")
                continue
            
            # Handle output format
            if isinstance(output, tuple):
                # Some models return (predictions, attention_weights, etc.)
                preds = output[0]
            else:
                preds = output
            
            # Convert to numpy
            if isinstance(preds, torch.Tensor):
                preds_np = preds.cpu().numpy()
            else:
                preds_np = np.array(preds)
            
            # Handle quantiles: if shape is (batch, horizon, quantiles), take median
            if preds_np.ndim == 3:
                median_idx = preds_np.shape[-1] // 2
                preds_np = preds_np[:, :, median_idx]
            
            all_predictions.append(preds_np)
            
            # Extract targets
            if y is not None:
                if isinstance(y, torch.Tensor):
                    targets_np = y.cpu().numpy()
                elif isinstance(y, (tuple, list)) and len(y) > 0:
                    # y is typically (decoder_target, encoder_target, ...)
                    if isinstance(y[0], torch.Tensor):
                        targets_np = y[0].cpu().numpy()
                    else:
                        targets_np = np.array(y[0])
                else:
                    targets_np = None
                
                if targets_np is not None:
                    all_targets.append(targets_np)
            
            if batch_count % 10 == 0:
                print(f"  Processed {batch_count} batches...")
    
    print(f"Total batches processed: {batch_count}\n")
    
    if not all_predictions or not all_targets:
        print("ERROR: No predictions or targets collected.")
        return
    
    # Flatten and align predictions and targets
    print("Aligning predictions and targets...")
    predictions_flat = np.concatenate([p.flatten() for p in all_predictions])
    targets_flat = np.concatenate([t.flatten() for t in all_targets])
    
    # Ensure same length
    min_len = min(len(predictions_flat), len(targets_flat))
    predictions_flat = predictions_flat[:min_len]
    targets_flat = targets_flat[:min_len]
    
    print(f"Total predictions: {len(predictions_flat)}")
    print(f"Total targets: {len(targets_flat)}\n")
    
    # Remove NaN/Inf
    valid = ~(np.isnan(predictions_flat) | np.isnan(targets_flat) | 
              np.isinf(predictions_flat) | np.isinf(targets_flat))
    preds_clean = predictions_flat[valid]
    targets_clean = targets_flat[valid]
    
    print(f"Valid samples (after removing NaN/Inf): {len(preds_clean)}\n")
    
    if len(preds_clean) == 0:
        print("ERROR: No valid predictions after cleaning.")
        return
    
    # Compute metrics
    print("Computing metrics...")
    print("-" * 80)
    
    rmse = np.sqrt(np.mean((preds_clean - targets_clean) ** 2))
    mae = np.mean(np.abs(preds_clean - targets_clean))
    mape = np.mean(np.abs((targets_clean - preds_clean) / (np.abs(targets_clean) + 1e-8))) * 100
    
    print(f"Overall Validation Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print("-" * 80)
    
    # Save sample predictions to CSV
    sample_size = min(100, len(preds_clean))
    sample_df = pd.DataFrame({
        "y_true": targets_clean[:sample_size],
        "y_pred": preds_clean[:sample_size],
        "error_abs": np.abs(targets_clean[:sample_size] - preds_clean[:sample_size]),
        "error_pct": np.abs((targets_clean[:sample_size] - preds_clean[:sample_size]) / 
                           (np.abs(targets_clean[:sample_size]) + 1e-8)) * 100
    })
    out_csv = ROOT / "tft_sample_predictions.csv"
    sample_df.to_csv(out_csv, index=False)
    print(f"\nSaved {sample_size} sample predictions to {out_csv}")
    
    # Summary stats
    print(f"\nPrediction Statistics:")
    print(f"  Predictions - Min: {preds_clean.min():.2f}, Max: {preds_clean.max():.2f}, Mean: {preds_clean.mean():.2f}")
    print(f"  Targets     - Min: {targets_clean.min():.2f}, Max: {targets_clean.max():.2f}, Mean: {targets_clean.mean():.2f}")
    
    print("\n✓ Evaluation complete.")

if __name__ == "__main__":
    main()