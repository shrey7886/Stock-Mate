# ============================================================
# TRAIN TEMPORAL FUSION TRANSFORMER (WITH / WITHOUT SENTIMENT)
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import torch
import lightning as L

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor


# ============================================================
# CONFIG
# ============================================================
USE_SENTIMENT = True           # TOGGLE: True = with sentiment, False = without
SEED = 42

DATA_PATH = Path("data/processed/tft_full_universe.parquet")
MODEL_DIR = Path("models/tft")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MAX_ENCODER_LENGTH = 60
MAX_PREDICTION_LENGTH = 7

BATCH_SIZE = 32
MAX_EPOCHS = 30
LEARNING_RATE = 1e-3

TRAIN_END_DATE = "2023-01-01"
VAL_END_DATE   = "2024-01-01"


# ============================================================
# REPRODUCIBILITY
# ============================================================
torch.manual_seed(SEED)
L.seed_everything(SEED, workers=True)
print(f"[INIT] Seed set to {SEED}")


# ============================================================
# LOAD DATA
# ============================================================
print("[LOAD] Loading dataset...")
df = pd.read_parquet(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["symbol", "time_idx"]).reset_index(drop=True)

print(f"[LOAD] Loaded data: {df.shape}")
print(f"[CONFIG] Using sentiment features: {USE_SENTIMENT}")


# ============================================================
# CLEAN NUMERICAL FEATURES (NO LEAKAGE)
# ============================================================
NUMERIC_COLS = [
    "returns_pct", "log_returns",
    "high_low_spread", "open_close_spread",
    "rolling_mean_7", "rolling_std_7", "rolling_std_30",
    "EMA_20", "EMA_50", "SMA_20",
    "RSI_14",
    "MACD", "MACD_signal", "MACD_hist",
    "ATR_14", "true_range",
    "bb_m", "bb_std", "bb_u", "bb_l",
    "bollinger_bandwidth",
    "volatility_20d", "volatility_60d",
]

# Add sentiment columns if requested
if USE_SENTIMENT:
    NUMERIC_COLS.extend([
        "sentiment_news_composite",
        "sentiment_social_score",
        "reddit_sentiment_mean",
        "reddit_post_volume",
    ])

# Filter to only existing columns
NUMERIC_COLS = [col for col in NUMERIC_COLS if col in df.columns]
print(f"[CLEAN] Processing {len(NUMERIC_COLS)} numeric columns")

# Replace inf -> NaN
df[NUMERIC_COLS] = df[NUMERIC_COLS].replace([np.inf, -np.inf], np.nan)

# Fill per symbol (forward fill, then backward fill)
print("[CLEAN] Filling missing values...")
df[NUMERIC_COLS] = (
    df.groupby("symbol", group_keys=False)[NUMERIC_COLS]
      .apply(lambda x: x.ffill().bfill())
)

# Final check
still_missing = df[NUMERIC_COLS].isna().sum().sum()
if still_missing > 0:
    print(f"[WARN] Still {still_missing} NaN values after filling")
    df = df.dropna(subset=NUMERIC_COLS)
    print(f"[CLEAN] Dropped rows with NaN - new shape: {df.shape}")


# ============================================================
# DROP ROWS WITHOUT FULL FUTURE HORIZON (TARGET SAFETY)
# ============================================================
TARGET_COLS = [
    "target_future_1", "target_future_2", "target_future_3",
    "target_future_4", "target_future_5", "target_future_6",
    "target_future_7",
]

# Check target columns exist
if not all(col in df.columns for col in TARGET_COLS):
    missing = [col for col in TARGET_COLS if col not in df.columns]
    print(f"[ERROR] Missing target columns: {missing}")
    exit(1)

# Replace inf -> NaN
df[TARGET_COLS] = df[TARGET_COLS].replace([float("inf"), float("-inf")], pd.NA)

# Drop rows with missing targets
before = len(df)
df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)
after = len(df)
print(f"[TARGET] Dropped {before - after} rows without full future horizon")


# ============================================================
# TRAIN / VAL / TEST SPLIT (TIME SAFE)
# ============================================================
train_df = df[df.date < TRAIN_END_DATE]
val_df   = df[(df.date >= TRAIN_END_DATE) & (df.date < VAL_END_DATE)]
test_df  = df[df.date >= VAL_END_DATE]

if train_df.empty or val_df.empty or test_df.empty:
    print("[ERROR] One or more splits is empty!")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Val: {len(val_df)} rows")
    print(f"  Test: {len(test_df)} rows")
    exit(1)

assert train_df.date.max() < val_df.date.min(), "Train/Val overlap"
assert val_df.date.max() < test_df.date.min(), "Val/Test overlap"

print(f"[SPLIT] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


# ============================================================
# FEATURE DEFINITIONS
# ============================================================
TIME_VARYING_KNOWN_REALS = [
    "time_idx", "day_of_week", "week_of_year", "month",
    "is_month_start", "is_month_end",
]

TIME_VARYING_UNKNOWN_REALS = [
    "open", "high", "low", "close", "volume",
    "returns_pct", "log_returns",
    "high_low_spread", "open_close_spread",
    "rolling_mean_7", "rolling_std_7", "rolling_std_30",
    "EMA_20", "EMA_50", "SMA_20",
    "RSI_14",
    "MACD", "MACD_signal", "MACD_hist",
    "ATR_14", "true_range",
    "bb_m", "bb_std", "bb_u", "bb_l",
    "bollinger_bandwidth",
    "volatility_20d", "volatility_60d",
]

if USE_SENTIMENT:
    TIME_VARYING_UNKNOWN_REALS.extend([
        "sentiment_news_composite",
        "sentiment_social_score",
        "reddit_sentiment_mean",
        "reddit_post_volume",
    ])

# Verify all features exist
missing_features = [f for f in TIME_VARYING_UNKNOWN_REALS if f not in train_df.columns]
if missing_features:
    print(f"[WARN] Missing features: {missing_features}")
    TIME_VARYING_UNKNOWN_REALS = [f for f in TIME_VARYING_UNKNOWN_REALS if f in train_df.columns]

print(f"[FEATURES] Using {len(TIME_VARYING_UNKNOWN_REALS)} time-varying unknown features")


# ============================================================
# TIME SERIES DATASET
# ============================================================
print("[DATASET] Creating TimeSeriesDataSet...")
try:
    training_ds = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=TARGET_COLS,
        group_ids=["symbol"],
        static_categoricals=["symbol"],
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        target_normalizer=MultiNormalizer(
            [GroupNormalizer(groups=["symbol"]) for _ in range(len(TARGET_COLS))]
        ),
        allow_missing_timesteps=False,
    )
    print("[DATASET] TimeSeriesDataSet created successfully")
except Exception as e:
    print(f"[ERROR] Failed to create TimeSeriesDataSet: {e}")
    exit(1)


# ============================================================
# VALIDATION / TEST DATASETS
# ============================================================
print("[DATASET] Creating validation and test datasets...")
try:
    val_ds = TimeSeriesDataSet.from_dataset(
        training_ds, val_df, predict=False, stop_randomization=True
    )
    test_ds = TimeSeriesDataSet.from_dataset(
        training_ds, test_df, predict=True, stop_randomization=True
    )
    print("[DATASET] Validation and test datasets created")
except Exception as e:
    print(f"[ERROR] Failed to create val/test datasets: {e}")
    exit(1)


# ============================================================
# DATALOADERS
# ============================================================
print("[LOADER] Creating dataloaders...")
train_loader = training_ds.to_dataloader(
    train=True, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

val_loader = val_ds.to_dataloader(
    train=False, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
print("[LOADER] Dataloaders ready")


# ============================================================
# MODEL
# ============================================================
print("[MODEL] Initializing Temporal Fusion Transformer...")
try:
    tft = TemporalFusionTransformer.from_dataset(
        training_ds,
        hidden_size=32,
        attention_head_size=4,
        hidden_continuous_size=16,
        dropout=0.1,
        learning_rate=LEARNING_RATE,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    print("[MODEL] TFT initialized successfully")
except Exception as e:
    print(f"[ERROR] Failed to initialize model: {e}")
    exit(1)


# ============================================================
# TRAINER
# ============================================================
print("[TRAINER] Setting up training callbacks...")
early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
lr_monitor = LearningRateMonitor(logging_interval="epoch")

trainer = Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="cpu",
    gradient_clip_val=0.1,
    callbacks=[early_stop, lr_monitor],
    log_every_n_steps=10,
    enable_progress_bar=True,
)
print("[TRAINER] Trainer configured")


# ============================================================
# TRAIN
# ============================================================
print("[TRAIN] Starting training...")
try:
    trainer.fit(
        tft,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    print("\n[SUCCESS] TFT training completed")
except Exception as e:
    print(f"[ERROR] Training failed: {e}")
    exit(1)


# ============================================================
# SAVE MODEL
# ============================================================
print("[SAVE] Saving model...")
try:
    suffix = "with_sentiment" if USE_SENTIMENT else "no_sentiment"
    checkpoint_path = MODEL_DIR / f"tft_{suffix}.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"[SAVE] Model saved to {checkpoint_path}")
except Exception as e:
    print(f"[ERROR] Failed to save model: {e}")
    exit(1)

print("\n[COMPLETE] Pipeline finished successfully!")
