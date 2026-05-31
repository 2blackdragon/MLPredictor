import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field

from app.config import settings
from app.prometheus_client import PrometheusClient
from app.feature_builder import FeatureBuilder
from app.predictor import CatBoostPredictor
import app.metrics_registry as reg

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Сохраняет предсказание с временем его создания."""
    value: float
    timestamp: float  # когда было сделано предсказание (unix)


@dataclass
class CollectorState:
    last_update_ts: float = 0.0
    errors_total: int = 0
    last_actual: float | None = None
    last_predicted: float | None = None
    last_features: dict = field(default_factory=dict)
    # Буфер предсказаний — хранит последние FORECAST_HORIZON предсказаний
    # Когда приходит новое значение, самое старое предсказание "созревает"
    predictions_buffer: deque = field(default_factory=deque)


state = CollectorState()

# Кольцевой буфер последних значений — чтобы не ходить в Prometheus лишний раз
_value_buffer: list[float] = []


async def _collect_once(
    prom: PrometheusClient,
    builder: FeatureBuilder,
    predictor: CatBoostPredictor,
) -> None:
    global _value_buffer

    # 1. Получаем историю из Prometheus
    values = await prom.fetch_recent_values(settings.HISTORY_POINTS)
    if values is None:
        state.errors_total += 1
        reg.collector_errors_total.set(state.errors_total)
        return

    _value_buffer = values
    actual = values[-1]
    current_ts = time.time()

    # 2. Строим вектор фичей
    try:
        features_df = builder.build(values)
    except ValueError as e:
        logger.warning("Feature build failed: %s", e)
        state.errors_total += 1
        reg.collector_errors_total.set(state.errors_total)
        return

    # 3. Предсказываем
    try:
        predicted = predictor.predict(features_df)
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        state.errors_total += 1
        reg.collector_errors_total.set(state.errors_total)
        return

    # 4. Проверяем, "созрело ли" самое старое предсказание
    #    Когда буфер достигает FORECAST_HORIZON записей, это значит,
    #    что текущий actual соответствует предсказанию, которое было FORECAST_HORIZON шагов назад
    if len(state.predictions_buffer) >= settings.FORECAST_HORIZON:
        old_record = state.predictions_buffer.popleft()
        error = abs(actual - old_record.value)
        reg.prediction_error.set(error)
        age_sec = current_ts - old_record.timestamp
        logger.info(
            "Prediction matured: predicted=%.2f%% actual=%.2f%% error=%.3f%% "
            "(age=%.0fs, ~%.1f min ago)",
            old_record.value,
            actual,
            error,
            age_sec,
            age_sec / 60,
        )

    # 5. Добавляем текущее предсказание в буфер
    state.predictions_buffer.append(PredictionRecord(predicted, current_ts))

    # 6. Обновляем Prometheus gauges
    reg.cpu_actual.set(actual)
    reg.cpu_predicted.set(predicted)

    # 7. Обновляем состояние
    state.last_update_ts = current_ts
    state.last_actual = actual
    state.last_predicted = predicted
    state.last_features = features_df.iloc[0].to_dict()

    reg.collector_lag_seconds.set(0)

    logger.debug(
        "actual=%.2f%% predicted=%.2f%% buffer_size=%d",
        actual,
        predicted,
        len(state.predictions_buffer),
    )


async def collector_loop(
    prom: PrometheusClient,
    builder: FeatureBuilder,
    predictor: CatBoostPredictor,
) -> None:
    """Бесконечный цикл, запускается как asyncio background task."""
    logger.info(
        "Collector started (interval=%ds, horizon=%d steps = %.0fs)",
        settings.SCRAPE_INTERVAL_SEC,
        settings.FORECAST_HORIZON,
        settings.FORECAST_HORIZON * settings.SCRAPE_INTERVAL_SEC,
    )
    while True:
        start = time.monotonic()
        try:
            await _collect_once(prom, builder, predictor)
        except Exception as e:
            logger.exception("Unexpected error in collector: %s", e)
            state.errors_total += 1
            reg.collector_errors_total.set(state.errors_total)

        # Обновляем lag — сколько секунд с последнего успешного обновления
        if state.last_update_ts:
            reg.collector_lag_seconds.set(time.time() - state.last_update_ts)

        # Ждём оставшееся время до следующего тика
        elapsed = time.monotonic() - start
        sleep_for = max(0.0, settings.SCRAPE_INTERVAL_SEC - elapsed)
        await asyncio.sleep(sleep_for)
