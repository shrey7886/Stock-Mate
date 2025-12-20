import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
import yaml
from pathlib import Path

# Get the absolute path to the config file
config_path = Path(__file__).parent.parent.parent / "ml_service" / "training" / "trainer_config.yaml"

try:
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
        if config_data and "TFT_CONFIG" in config_data:
            CONFIG = config_data["TFT_CONFIG"]
        else:
            # Default config if not found in YAML
            CONFIG = {
                "encoder_length": 60,
                "prediction_length": 30,
                "known_reals": ["day_of_week", "week_of_year", "month"],
                "unknown_reals": ["returns_pct", "volume"]
            }
except FileNotFoundError:
    # Default config if file doesn't exist
    CONFIG = {
        "encoder_length": 60,
        "prediction_length": 30,
        "known_reals": ["day_of_week", "week_of_year", "month"],
        "unknown_reals": ["returns_pct", "volume"]
    }


class TFTDatasetBuilder:
    def __init__(self, train_df_path, val_df_path):
        self.train_df = pd.read_parquet(train_df_path)
        self.val_df = pd.read_parquet(val_df_path)

    def build(self):
        dataset = TimeSeriesDataSet(
            self.train_df,
            time_idx="time_idx",
            target="close",
            group_ids=["symbol"],
            min_encoder_length=CONFIG["encoder_length"],
            max_encoder_length=CONFIG["encoder_length"],
            max_prediction_length=CONFIG["prediction_length"],
            static_categoricals=["symbol"],
            time_varying_known_reals=CONFIG["known_reals"],
            time_varying_unknown_reals=CONFIG["unknown_reals"],
            add_relative_time_idx=True,
            add_target_scales=True,
        )

        validation = dataset.from_dataset(dataset, self.val_df)

        return dataset, validation
