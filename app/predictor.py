import logging
import pandas as pd
from catboost import CatBoostRegressor

from app.config import settings

logger = logging.getLogger(__name__)


class CatBoostPredictor:
    """Обёртка над обученной CatBoost моделью."""

    def __init__(self):
        self.model: CatBoostRegressor | None = None
        self._load()

    def _load(self):
        try:
            model = CatBoostRegressor()
            model.load_model(settings.MODEL_PATH)
            self.model = model
            logger.info(
                "CatBoost model loaded from '%s'", settings.MODEL_PATH
            )
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise

    def predict(self, features: pd.DataFrame) -> float:
        """
        Принимает DataFrame из одной строки (выход FeatureBuilder).
        Возвращает предсказанный CPU% через FORECAST_HORIZON × 15с = 2.5 мин.
        Значение зажато в [0, 100].
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        raw: float = float(self.model.predict(features)[0])
        return max(0.0, min(100.0, raw))
