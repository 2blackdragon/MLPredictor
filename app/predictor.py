import logging
import pandas as pd
from catboost import CatBoostRegressor
from app.config import MetricConfig

logger = logging.getLogger(__name__)


class MetricPredictor:
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

    def predict(self, features: pd.DataFrame) -> float:
        if self.model is None:
            raise RuntimeError(f"Model for '{self.metric_type}' is not loaded")

        raw: float = float(self.model.predict(features)[0])

        if self.metric_type == "cpu":
            return max(0.0, min(100.0, raw))
        elif self.metric_type == "error_rate":
            return max(0.0, min(1.0, raw))
        else:
            return max(0.0, raw)
