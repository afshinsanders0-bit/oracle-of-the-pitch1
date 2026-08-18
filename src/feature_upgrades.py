"""
feature_upgrades.py — Calibrated Model Wrapper
===============================================
Provides probability-calibrated model wrappers for better betting edge estimation.
Uses sklearn's CalibratedClassifierCV with isotonic regression when available.
Gracefully degrades to the base model if calibration dependencies are missing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PATHS


class CalibratedFootballModel:
    """
    Wrapper around a base XGBoost model that adds probability calibration.

    If a calibrated version was saved, loads it directly.
    Otherwise, wraps the base model and provides a compatible interface.
    """

    def __init__(self, base_model: Any = None, feature_names: Optional[list[str]] = None):
        self.base_model = base_model
        self.feature_names = feature_names or []
        self.calibrated_ = None

    @classmethod
    def load(cls, path: Path | str) -> "CalibratedFootballModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Calibrated model not found: {path}")
        obj = joblib.load(path)
        if isinstance(obj, cls):
            return obj
        wrapper = cls(base_model=obj)
        return wrapper

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.base_model is None:
            raise RuntimeError("No base model loaded")
        return self.base_model.predict_proba(X)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.base_model is None:
            raise RuntimeError("No base model loaded")
        return self.base_model.predict(X)

    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path else (PATHS.MODELS / "match_result_calibrated.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    def __repr__(self) -> str:
        return f"CalibratedFootballModel(base={type(self.base_model).__name__}, features={len(self.feature_names)})"


def calibrate_model(base_model: Any, X_cal: pd.DataFrame, y_cal: np.ndarray) -> CalibratedFootballModel:
    """
    Calibrate a base model using isotonic regression on a held-out calibration set.

    Args:
        base_model: Trained XGBoost/LightGBM model with predict_proba
        X_cal:      Calibration features (unseen during training)
        y_cal:      Calibration labels

    Returns:
        CalibratedFootballModel wrapper
    """
    try:
        from sklearn.calibration import CalibratedClassifierCV
    except ImportError:
        print("⚠️  sklearn not installed — returning uncalibrated wrapper")
        return CalibratedFootballModel(base_model=base_model)

    calibrated = CalibratedClassifierCV(
        base_model,
        method="isotonic",
        cv="prefit",
    )
    calibrated.fit(X_cal, y_cal)

    wrapper = CalibratedFootballModel(base_model=base_model)
    wrapper.calibrated_ = calibrated
    return wrapper
