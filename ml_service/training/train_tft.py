# ml_service/training/train_tft.py
"""
Train a Temporal Fusion Transformer (TFT) using pytorch-forecasting.

Usage:
    python -m ml_service.training.train_tft

Outputs:
    - Trained model checkpoint saved to ml_service/models/saved_models/tft/
"""

from pathlib import Path
import pandas as pd
import numpy as np
import os
import torch

from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

# Config
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "ml_service" / "models" / "saved_models" / "tft"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_EPOCHS = int(os.getenv("TFT_MAX_EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("TFT_BATCH_SIZE", "64"))
HIDDEN_SIZE = int(os.getenv("TFT_HIDDEN_SIZE", "16"))
LEARNING_RATE = float(os.getenv("TFT_LR", "1e-3"))
RANDOM_SEED = 42

def load_processed_universe() -> pd.DataFrame:
    files = sorted(PROCESSED_DIR.glob("*_tft.parquet"))
    if not files:
        raise FileNotFoundError(f"No processed tft parquet files found in {PROCESSED_DIR}")
    dfs = [pd.read_parquet(p) for p in files]
    df = pd.concat(dfs, ignore_index=True)
    # ensure timestamp exists as datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    # ensure time_idx exists per symbol
    if "time_idx" not in df.columns:
        df = df.sort_values(["symbol", "timestamp"])
        df["time_idx"] = df.groupby("symbol").cumcount()
    return df

def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace inf with NaN and forward/back-fill numeric columns per symbol using transform
    (keeps index alignment). Add indicator columns for imputed values.
    """
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("time_idx", "symbol")]

    for col in numeric_cols:
        na_col = f"{col}_was_na"
        df[na_col] = df[col].isna().astype(int)
        # transform preserves index alignment
        df[col] = df.groupby("symbol")[col].transform(lambda s: s.ffill().bfill())
        df[col] = df[col].fillna(0.0)
    return df

def prepare_datasets(df: pd.DataFrame, max_encoder_length=15, max_prediction_length=3):
    """
    Prepare a single TimeSeriesDataSet from df.
    PyTorch Lightning will handle train/val split internally.
    Uses shorter encoder/decoder lengths to fit more windows in small datasets.
    """
    df = df.copy()
    
    # Sort by symbol and timestamp
    if "timestamp" in df.columns:
        df = df.sort_values(["symbol", "timestamp"])
    
    if "time_idx" not in df.columns:
        df["time_idx"] = df.groupby("symbol").cumcount()

    # Clean numerics
    df = clean_numeric_columns(df)

    # Define target and variables
    target = "target_future_1"
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found in processed data. Columns: {df.columns.tolist()}")

    # DROP ROWS WITH NaN TARGETS (critical for training)
    print(f"Rows before dropping NaN targets: {len(df)}")
    df = df.dropna(subset=[target])
    print(f"Rows after dropping NaN targets: {len(df)}")
    
    if len(df) == 0:
        raise ValueError("All data has NaN targets. Check add_targets() in dataset_generator.")

    # Choose a set of real variables (prioritize sentiment + core)
    real_vars = []
    for c in ["close", "returns_pct", "volatility_20d", "volatility_60d", "sentiment_news_composite", "sentiment_social_score"]:
        if c in df.columns:
            real_vars.append(c)
    # append some other numeric cols if present
    potential_reals = [c for c in df.select_dtypes(include=["float64", "int64"]).columns 
                       if c not in real_vars and c not in ("time_idx",) and not c.endswith("_was_na")]
    for c in potential_reals:
        if len(real_vars) >= 12:
            break
        if c != target:
            real_vars.append(c)

    static_categoricals = ["symbol"]
    time_varying_known_reals = ["time_idx"]
    time_varying_unknown_reals = real_vars

    print(f"Using {len(real_vars)} real variables: {real_vars}")
    
    # Drop helper columns like 'date' that shouldn't be in the dataset
    drop_cols = [c for c in df.columns if c in ["date"] or c.startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
        print(f"Dropped helper columns: {drop_cols}")

    # Build single TimeSeriesDataSet on full df
    try:
        dataset = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=target,
            group_ids=["symbol"],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=static_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=["symbol"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
    except Exception as e:
        print(f"ERROR building TimeSeriesDataSet: {e}")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"DataFrame dtypes:\n{df.dtypes}")
        raise

    return dataset

class LightningModelAdapter(LightningModule):
    """
    Adapter to wrap pytorch-forecasting models so Trainer sees a LightningModule.
    Ensures the wrapped model has `trainer`, `log`, `log_dict`, and `save_hyperparameters`
    attached before any of its hook methods are called.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def _ensure_wrapped_attached(self):
        """
        Attach trainer and logging methods to the wrapped model.
        This is called right before delegating to any wrapped hook.
        """
        try:
            # attach trainer reference from adapter (Trainer sets adapter.trainer)
            if getattr(self.model, "trainer", None) is None and getattr(self, "trainer", None) is not None:
                self.model.trainer = self.trainer

            # proxy logging methods
            if not hasattr(self.model, "log"):
                self.model.log = self.log  # type: ignore
            if not hasattr(self.model, "log_dict"):
                self.model.log_dict = self.log_dict  # type: ignore

            # ensure save_hyperparameters exists on wrapped model
            if not hasattr(self.model, "save_hyperparameters"):
                self.model.save_hyperparameters = self.save_hyperparameters  # type: ignore
        except Exception:
            # keep robust: don't fail adapter attachment if any attribute set fails
            pass

    def on_fit_start(self) -> None:
        self._ensure_wrapped_attached()

    def on_train_start(self) -> None:
        self._ensure_wrapped_attached()

    def on_validation_start(self) -> None:
        self._ensure_wrapped_attached()

    def on_test_start(self) -> None:
        self._ensure_wrapped_attached()

    def forward(self, *args, **kwargs):
        self._ensure_wrapped_attached()
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        # ensure the wrapped model has trainer and logging attached before calling
        self._ensure_wrapped_attached()
        return self.model.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self._ensure_wrapped_attached()
        return self.model.validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self._ensure_wrapped_attached()
        if hasattr(self.model, "test_step"):
            return self.model.test_step(batch, batch_idx)
        return None

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self._ensure_wrapped_attached()
        if hasattr(self.model, "predict_step"):
            return self.model.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        return self.model(batch)

    def configure_optimizers(self):
        self._ensure_wrapped_attached()
        return self.model.configure_optimizers()

def main():
    torch.manual_seed(RANDOM_SEED)
    print("Loading processed data...")
    df = load_processed_universe()
    print("Rows:", len(df), "Symbols:", df["symbol"].nunique())
    print("Columns:", df.columns.tolist())

    print("\nPreparing dataset...")
    dataset = prepare_datasets(df)
    print("Dataset size (windows):", len(dataset))
    
    if len(dataset) == 0:
        raise ValueError("TimeSeriesDataSet is empty. Check encoder/decoder lengths or data size.")

    # dataloaders
    train_dataloader = dataset.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_dataloader = dataset.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=0)

    # create model with QuantileLoss (required for TFT)
    tft = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # attempt to silence hyperparameter checkpoint warnings
    try:
        tft.save_hyperparameters(ignore=["loss", "logging_metrics"])
    except Exception:
        # not critical; continue
        pass

    # wrap in adapter if Trainer does not recognize the model as LightningModule
    adapter_needed = not isinstance(tft, LightningModule)
    if adapter_needed:
        print("[info] Wrapping TemporalFusionTransformer in LightningModelAdapter for compatibility.")
        model_to_train = LightningModelAdapter(tft)
    else:
        model_to_train = tft

    # logging & checkpointing
    logger = CSVLogger("lightning_logs", name="tft")
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(OUTPUT_DIR),
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # Use accelerator and devices instead of deprecated gpus parameter
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        gradient_clip_val=0.1,
        enable_model_summary=True,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        default_root_dir=str(OUTPUT_DIR),
    )

    print("Starting training...")
    trainer.fit(model_to_train, train_dataloader, val_dataloader)

    print("Training completed. Best model saved to:", OUTPUT_DIR)
    print("You can load the model with TemporalFusionTransformer.load_from_checkpoint(path)")

if __name__ == "__main__":
    main()