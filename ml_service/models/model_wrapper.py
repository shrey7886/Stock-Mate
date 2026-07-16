from __future__ import annotations

from pathlib import Path


def resolve_model_dir(model_name: str) -> Path:
	model_name = model_name.lower().strip()
	if model_name == "patchtst":
		return Path("ml_service/models/saved_models/patchtst")
	if model_name == "tft":
		return Path("models/tft")
	raise ValueError(f"Unsupported model: {model_name}")
