import asyncio
import time
from collections import deque
import logging
from app.influx_client import InfluxWriter
from app.feature_builder import FeatureBuilder
from app.predictor import CatBoostPredictor
from app.config import settings

logger = logging.getLogger(__name__)


class CollectorState:
    def __init__(self):
        self.predictions_buffer: deque = deque(maxlen=500)
        self.errors_total = 0
        self.last_update_ts: float | None = None
        self.last_actual: float | None = None
        self.last_predicted: float | None = None
        self.last_features: list[float] = []

state = CollectorState()
influx = InfluxWriter()


async def _collect_once(
    prom_client,
    builder: FeatureBuilder,
    predictor: CatBoostPredictor,
) -> None:
    # 1. Получаем актуальные данные
    values = await prom_client.fetch_recent_values(settings.HISTORY_POINTS)
    if values is None or len(values) < settings.HISTORY_POINTS:
        logger.warning("Not enough data")
        state.errors_total += 1
        return

    actual = values[-1]
    current_ts = time.time()

    # 2. Предсказание
    try:
        features_df = builder.build_features(values)
        features_list = features_df.iloc[0].tolist()
        predicted = predictor.predict(features_df)
    except Exception as e:
        logger.error("Prediction error: %s", e)
        state.errors_total += 1
        return

    horizon_seconds = settings.FORECAST_HORIZON * settings.SCRAPE_INTERVAL_SEC

    # 3. Записываем в InfluxDB (predicted с будущим временем!)
    influx.write_cpu_metrics(actual, predicted, horizon_seconds)

    # 4. Оценка ошибки
    if len(state.predictions_buffer) >= settings.FORECAST_HORIZON:
        old_predicted = state.predictions_buffer.popleft()
        error = abs(actual - old_predicted)
        influx.write_error(error)
        logger.info(f"Prediction matured | error={error:.3f}%")

    # Добавляем текущее предсказание в буфер
    state.predictions_buffer.append(predicted)

    # Лаг
    lag = time.time() - current_ts
    influx.write_lag(lag)

    # Обновляем состояние для health endpoint
    state.last_update_ts = current_ts
    state.last_actual = actual
    state.last_predicted = predicted
    state.last_features = features_list

    logger.info(
        f"actual={actual:.2f}% | predicted={predicted:.2f}% (+{horizon_seconds}s)"
    )


async def start_collector(prom_client, builder, predictor):
    while True:
        try:
            await _collect_once(prom_client, builder, predictor)
        except Exception as e:
            logger.error("Collector error: %s", e)
            state.errors_total += 1

        await asyncio.sleep(settings.SCRAPE_INTERVAL_SEC)
