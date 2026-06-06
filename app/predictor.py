import logging

import joblib
import pandas as pd
from catboost import CatBoostRegressor

from app.config import MetricConfig

logger = logging.getLogger(__name__)


class MetricPredictor:
    """CatBoost predictor (RPS / error_rate)."""

    def __init__(self, metric_config: MetricConfig):
        self.metric_config = metric_config
        self.metric_type = metric_config.metric_type
        self.model: CatBoostRegressor | None = None
        self._load()

    def _load(self):
        try:
            model = CatBoostRegressor()
            model.load_model(self.metric_config.model_path)
            self.model = model
            logger.info(
                "CatBoost model loaded for '%s' from '%s'",
                self.metric_type,
                self.metric_config.model_path,
            )
        except Exception as e:
            logger.error("Failed to load %s model: %s", self.metric_type, e)
            raise

    def _clip(self, raw: float) -> float:
        if self.metric_type == "cpu":
            return max(0.0, min(100.0, raw))
        if self.metric_type == "error_rate":
            return max(0.0, min(1.0, raw))
        return max(0.0, raw)

    def predict(self, features: pd.DataFrame) -> float:
        if self.model is None:
            raise RuntimeError(f"Model for '{self.metric_type}' is not loaded")
        raw = float(self.model.predict(features)[0])
        return self._clip(raw)


class RidgePredictor(MetricPredictor):
    """
    Ridge predictor for CPU (Yandex Handbook §10.3, walk-forward Scheme 2).

    Loads a single joblib bundle produced by `training/cpu_walkforward.ipynb`:
        {
            'model':   Ridge,
            'scaler':  StandardScaler,
            'feature_cols': [...],
            'meta':    {...},
        }
    """

    def __init__(self, metric_config: MetricConfig):
        self.metric_config = metric_config
        self.metric_type = metric_config.metric_type
        self.model = None
        self.scaler = None
        self.feature_cols: list[str] = []
        self.meta: dict = {}
        self._load()

    def _load(self):
        try:
            bundle = joblib.load(self.metric_config.model_path)
        except Exception as e:
            logger.error("Failed to load %s ridge bundle: %s", self.metric_type, e)
            raise

        if not isinstance(bundle, dict) or "model" not in bundle or "scaler" not in bundle:
            raise RuntimeError(
                f"Ridge bundle for '{self.metric_type}' is malformed: "
                f"expected dict with 'model' and 'scaler'"
            )

        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_cols = bundle.get("feature_cols", []) or self.metric_config.feature_cols
        self.meta = bundle.get("meta", {}) or {}

        logger.info(
            "Ridge model loaded for '%s' from '%s' (horizon=%ss, features=%d)",
            self.metric_type,
            self.metric_config.model_path,
            self.meta.get(
                "forecast_horizon_seconds",
                self.metric_config.forecast_horizon_seconds,
            ),
            len(self.feature_cols),
        )

    def predict(self, features: pd.DataFrame) -> float:
        if self.model is None or self.scaler is None:
            raise RuntimeError(f"Ridge model for '{self.metric_type}' is not loaded")

        # Enforce exact column order used during training.
        if self.feature_cols:
            missing = [c for c in self.feature_cols if c not in features.columns]
            if missing:
                raise ValueError(f"Missing features for ridge inference: {missing}")
            features = features[self.feature_cols]

        scaled = self.scaler.transform(features.values)
        raw = float(self.model.predict(scaled)[0])
        return self._clip(raw)


def build_predictor(metric_config: MetricConfig) -> MetricPredictor:
    if metric_config.model_type == "ridge":
        return RidgePredictor(metric_config)
    return MetricPredictor(metric_config)
