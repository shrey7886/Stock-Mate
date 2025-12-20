import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_service.models.data.feature_engineering import TFTFeatureEngineer

# Use absolute paths - dataset_generator is at ml_service/models/data/dataset_generator.py
# So parent.parent.parent takes us to ml_service, need to go up one more to project root
project_root = Path(__file__).parent.parent.parent.parent
DATASET_PATH = project_root / "data_pipeline" / "final_dataset.csv"
OUTPUT_DIR = project_root / "ml_service" / "models" / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class TFTDatasetGenerator:
    def __init__(self):
        self.engineer = TFTFeatureEngineer()

    def load_dataset(self):
        if not DATASET_PATH.exists():
            raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run data pipeline first with:\npython data_pipeline/scheduler/cron_jobs.py")
        df = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
        return df

    def split(self, df, train_ratio=0.8, val_ratio=0.1):
        symbols = df["symbol"].unique()
        train_dfs, val_dfs, test_dfs = [], [], []

        for sym in symbols:
            sub = df[df["symbol"] == sym].sort_values("time_idx")
            n = len(sub)

            train_dfs.append(sub.iloc[:int(n * train_ratio)])
            val_dfs.append(sub.iloc[int(n * train_ratio):int(n * (train_ratio + val_ratio))])
            test_dfs.append(sub.iloc[int(n * (train_ratio + val_ratio)):])

        return (
            pd.concat(train_dfs, ignore_index=True),
            pd.concat(val_dfs, ignore_index=True),
            pd.concat(test_dfs, ignore_index=True)
        )

    def generate(self):
        df = self.load_dataset()

        # Add time index
        df = self.engineer.add_time_idx(df)

        # Encode symbol
        df = self.engineer.encode_static_categoricals(df)

        # Scale numeric features
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        # Remove 'symbol' if it got encoded and is in numeric cols
        numeric_cols = [col for col in numeric_cols if col not in ['symbol', 'time_idx']]
        if numeric_cols:
            df = self.engineer.scale_features(df, numeric_cols)

        # Split
        train_df, val_df, test_df = self.split(df)

        # Save
        train_df.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
        val_df.to_parquet(OUTPUT_DIR / "val.parquet", index=False)
        test_df.to_parquet(OUTPUT_DIR / "test.parquet", index=False)

        print(f"✔ TFT dataset generated → {OUTPUT_DIR}")
