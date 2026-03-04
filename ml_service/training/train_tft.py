from pathlib import Path
import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor


# ===============================
# CONFIG
# ===============================
DATA_PATH = Path("data/processed/tft_full_universe.parquet")

MAX_ENCODER_LENGTH = 60
MAX_PREDICTION_LENGTH = 7

BATCH_SIZE = 64
MAX_EPOCHS = 30
LEARNING_RATE = 1e-3

TRAIN_END_DATE = "2023-01-01"
VAL_END_DATE = "2024-01-01"


# ===============================
# LOAD DATA
# ===============================
df = pd.read_parquet(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["symbol", "time_idx"]).reset_index(drop=True)


# ===============================
# TRAIN / VAL / TEST SPLIT
# ===============================
train_df = df[df.date < TRAIN_END_DATE]
val_df = df[(df.date >= TRAIN_END_DATE) & (df.date < VAL_END_DATE)]
test_df = df[df.date >= VAL_END_DATE]

assert train_df.date.max() < val_df.date.min()
assert val_df.date.max() < test_df.date.min()

print("✔ Time-based split verified")
print(f"Train rows: {len(train_df)}")
print(f"Val rows  : {len(val_df)}")
print(f"Test rows : {len(test_df)}")


# ===============================
# TIME SERIES DATASET
# ===============================
training_ds = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target=[
        "target_future_1",
        "target_future_2",
        "target_future_3",
        "target_future_4",
        "target_future_5",
        "target_future_6",
        "target_future_7",
    ],
    group_ids=["symbol"],
    static_categoricals=["symbol"],

    time_varying_known_reals=[
        "time_idx",
        "day_of_week",
        "week_of_year",
        "month",
        "is_month_start",
        "is_month_end",
    ],

    time_varying_unknown_reals=[
        "open", "high", "low", "close", "volume",
        "high_low_spread", "open_close_spread",
        "rolling_mean_7", "rolling_std_7", "rolling_std_30",
        "EMA_20", "EMA_50", "SMA_20",
        "RSI_14",
        "MACD", "MACD_signal", "MACD_hist",
        "ATR_14", "true_range",
        "bollinger_bandwidth",
        "volatility_20d", "volatility_60d",
        #"sentiment_news_composite",
        #"sentiment_social_score",
    ],

    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PREDICTION_LENGTH,

    # Multiple targets require MultiNormalizer; each is a log return
    target_normalizer=MultiNormalizer(
        [GroupNormalizer(groups=["symbol"]) for _ in range(7)]
    ),

    allow_missing_timesteps=False,
)


# ===============================
# VALIDATION / TEST DATASETS
# ===============================
val_ds = TimeSeriesDataSet.from_dataset(
    training_ds, val_df, predict=False, stop_randomization=True
)

test_ds = TimeSeriesDataSet.from_dataset(
    training_ds, test_df, predict=True, stop_randomization=True
)


# ===============================
# DATALOADERS (NO SHUFFLING)
# ===============================
train_loader = training_ds.to_dataloader(
    train=True, batch_size=BATCH_SIZE, shuffle=False
)

val_loader = val_ds.to_dataloader(
    train=False, batch_size=BATCH_SIZE, shuffle=False
)


# ===============================
# MODEL
# ===============================
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


# ===============================
# TRAINER
# ===============================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min"
)

lr_monitor = LearningRateMonitor(logging_interval="epoch")

trainer = Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="cpu",
    gradient_clip_val=0.1,
    callbacks=[early_stop, lr_monitor],
)


# ===============================
# TRAIN
# ===============================
trainer.fit(
    tft,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)

print("\n✅ TFT training completed (leakage-safe)")


# ===============================
# SAVE MODEL
# ===============================
MODEL_DIR = Path("models/tft")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

trainer.save_checkpoint(MODEL_DIR / "tft_leakage_safe_no_sentiment.ckpt")
print("✔ Model saved to models/tft/tft_leakage_safe_no_sentiment.ckpt")