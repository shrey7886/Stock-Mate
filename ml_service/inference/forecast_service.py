from __future__ import annotations

from pathlib import Path

import pandas as pd
from neuralforecast import NeuralForecast

from ml_service.models.patchtst_pipeline import load_patchtst_frame


def load_patchtst(model_dir: str = "ml_service/models/saved_models/patchtst") -> NeuralForecast:
	"""Load a trained PatchTST NeuralForecast bundle from disk."""
	return NeuralForecast.load(path=str(Path(model_dir)), weights_only=False)


def predict_next_horizon(
	model_dir: str = "ml_service/models/saved_models/patchtst",
	data_path: str = "data/processed/tft_full_universe.parquet",
	target_col: str = "target_future_1",
) -> pd.DataFrame:
	"""Generate the next forecast horizon for each symbol from latest available history."""
	nf = load_patchtst(model_dir=model_dir)
	bundle = load_patchtst_frame(data_path=Path(data_path), target_col=target_col)
	forecasts = nf.predict(df=bundle.frame)
	return forecasts.sort_values(["unique_id", "ds"]).reset_index(drop=True)
