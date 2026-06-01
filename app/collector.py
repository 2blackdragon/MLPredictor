import asyncio
import time
from collections import deque
import logging
import math

from app.influx_client import InfluxWriter
from app.feature_builder import FeatureBuilder
from app.predictor import MetricPredictor
from app.prometheus_client import PrometheusClient
from app.config import settings

logger = logging.getLogger(__name__)


class MetricCollectorState:
    def __init__(self, metric_type: str):
        self.metric_type = metric_type
        self.predictions_buffer: deque = deque(maxlen=500)
        self.actual_history: deque = deque(maxlen=100)
        self.predicted_history: deque = deque(maxlen=100)
        self.errors_total = 0
        self.last_update_ts: float | None = None
        self.last_actual: float | None = None
        self.last_predicted: float | None = None
        self.last_features: list[float] = []

    def calculate_ml_metrics(self) -> dict[str, float] | None:
        n = len(self.actual_history)
        if n < 15:
            return None

        y_true = list(self.actual_history)
        y_pred = list(self.predicted_history)

        mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
        mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n
        rmse = math.sqrt(mse)

        mean_true = sum(y_true) / n
        ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
        ss_tot = sum((t - mean_true) ** 2 for t in y_true)

        if ss_tot < 1e-5:
            r2 = 1.0 if ss_res < 1e-5 else 0.0
        else:
            r2 = 1.0 - (ss_res / ss_tot)

        r2 = max(r2, -1.0)

        return {"r2": r2, "mae": mae, "rmse": rmse}


class HandlerCollectorState:
    def __init__(self, handler: str):
        self.handler = handler
        self.predictions_buffer: deque = deque(maxlen=500)
        self.actual_history: deque = deque(maxlen=100)
        self.predicted_history: deque = deque(maxlen=100)
        self.errors_total = 0
        self.last_update_ts: float | None = None
        self.last_actual: float | None = None
        self.last_predicted: float | None = None
        self.last_features: list[float] = []

    def calculate_ml_metrics(self) -> dict[str, float] | None:
        n = len(self.actual_history)
        if n < 15:
            return None

        y_true = list(self.actual_history)
        y_pred = list(self.predicted_history)

        mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
        mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n
        rmse = math.sqrt(mse)

        mean_true = sum(y_true) / n
        ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
        ss_tot = sum((t - mean_true) ** 2 for t in y_true)

        if ss_tot < 1e-5:
            r2 = 1.0 if ss_res < 1e-5 else 0.0
        else:
            r2 = 1.0 - (ss_res / ss_tot)

        r2 = max(r2, -1.0)

        return {"r2": r2, "mae": mae, "rmse": rmse}


metric_states = {
    "cpu": MetricCollectorState("cpu"),
    "rps": MetricCollectorState("rps"),
    "error_rate": MetricCollectorState("error_rate"),
}

handler_states: dict[str, HandlerCollectorState] = {}

influx = InfluxWriter()


async def _collect_once(
    metric_type: str,
    prom_client: PrometheusClient,
    builder: FeatureBuilder,
    predictor: MetricPredictor,
) -> None:
    state = metric_states[metric_type]
    metric_config = predictor.metric_config
    current_ts = time.time()
    horizon_seconds = settings.FORECAST_HORIZON * settings.SCRAPE_INTERVAL_SEC

    if metric_type == "rps":
        await _collect_rps_per_handler(prom_client, builder, predictor, metric_config, horizon_seconds)
    else:
        await _collect_single_metric(metric_type, prom_client, builder, predictor, state, horizon_seconds, current_ts)


async def _collect_single_metric(
    metric_type: str,
    prom_client: PrometheusClient,
    builder: FeatureBuilder,
    predictor: MetricPredictor,
    state: MetricCollectorState,
    horizon_seconds: int,
    current_ts: float,
) -> None:
    metric_config = predictor.metric_config

    values = await prom_client.fetch_recent_values(
        metric_config.prometheus_query,
        settings.HISTORY_POINTS,
        settings.SCRAPE_INTERVAL_SEC,
    )
    if values is None or len(values) < settings.HISTORY_POINTS:
        logger.warning("Not enough data for %s", metric_type)
        state.errors_total += 1
        return

    actual = values[-1]

    try:
        features_df = builder.build_features(values)
        features_list = features_df.iloc[0].tolist()
        predicted = predictor.predict(features_df)
    except Exception as e:
        logger.error("Prediction error for %s: %s", metric_type, e)
        state.errors_total += 1
        return

    influx.write_metric(
        metric_type,
        metric_config.influx_measurement,
        actual,
        predicted,
        horizon_seconds,
    )

    if len(state.predictions_buffer) >= settings.FORECAST_HORIZON:
        old_predicted = state.predictions_buffer.popleft()
        error = abs(actual - old_predicted)
        influx.write_error(metric_type, error)

        state.actual_history.append(actual)
        state.predicted_history.append(old_predicted)

        metrics = state.calculate_ml_metrics()
        if metrics:
            influx.write_ml_metrics(
                metric_type,
                r2=metrics["r2"],
                mae=metrics["mae"],
                rmse=metrics["rmse"],
            )
            logger.info(
                f"[{metric_type}] Metrics updated (window={len(state.actual_history)}) | "
                f"R²={metrics['r2']:.3f} | MAE={metrics['mae']:.2f} | RMSE={metrics['rmse']:.2f}"
            )

        logger.info(f"[{metric_type}] Prediction matured | error={error:.3f}")

    state.predictions_buffer.append(predicted)

    lag = time.time() - current_ts
    influx.write_lag(metric_type, lag)

    state.last_update_ts = current_ts
    state.last_actual = actual
    state.last_predicted = predicted
    state.last_features = features_list

    logger.info(f"[{metric_type}] actual={actual:.2f} | predicted={predicted:.2f} (+{horizon_seconds}s)")


async def _collect_rps_per_handler(
    prom_client: PrometheusClient,
    builder: FeatureBuilder,
    predictor: MetricPredictor,
    metric_config,
    horizon_seconds: int,
) -> None:
    current_ts = time.time()

    handler_values = await prom_client.fetch_recent_values_per_series(
        metric_config.prometheus_query,
        settings.HISTORY_POINTS,
        settings.SCRAPE_INTERVAL_SEC,
        label_name="handler",
    )

    if not handler_values:
        logger.warning("No handler data for RPS")
        metric_states["rps"].errors_total += 1
        return

    for handler, values in handler_values.items():
        if len(values) < settings.HISTORY_POINTS:
            logger.warning("Not enough data for handler %s", handler)
            continue

        handler_encoding = metric_config.get_handler_encoding(handler)
        if handler_encoding is None:
            logger.warning("Unknown handler: %s", handler)
            continue

        actual = values[-1]

        try:
            features_df = builder.build_features(values, categorical_value=handler_encoding)
            features_list = features_df.iloc[0].tolist()
            predicted = predictor.predict(features_df)
        except Exception as e:
            logger.error("Prediction error for RPS handler %s: %s", handler, e)
            continue

        influx.write_metric(
            "rps",
            metric_config.influx_measurement,
            actual,
            predicted,
            horizon_seconds,
            handler=handler,
        )

        if handler not in handler_states:
            handler_states[handler] = HandlerCollectorState(handler)

        state = handler_states[handler]

        if len(state.predictions_buffer) >= settings.FORECAST_HORIZON:
            old_predicted = state.predictions_buffer.popleft()
            error = abs(actual - old_predicted)
            influx.write_error("rps", error, handler=handler)

            state.actual_history.append(actual)
            state.predicted_history.append(old_predicted)

            metrics = state.calculate_ml_metrics()
            if metrics:
                influx.write_ml_metrics(
                    "rps",
                    r2=metrics["r2"],
                    mae=metrics["mae"],
                    rmse=metrics["rmse"],
                    handler=handler,
                )
                logger.info(
                    f"[rps:{handler}] Metrics updated (window={len(state.actual_history)}) | "
                    f"R²={metrics['r2']:.3f} | MAE={metrics['mae']:.2f} | RMSE={metrics['rmse']:.2f}"
                )

            logger.info(f"[rps:{handler}] Prediction matured | error={error:.3f}")

        state.predictions_buffer.append(predicted)

        state.last_update_ts = current_ts
        state.last_actual = actual
        state.last_predicted = predicted
        state.last_features = features_list

        logger.info(f"[rps:{handler}] actual={actual:.2f} | predicted={predicted:.2f} (+{horizon_seconds}s)")

    lag = time.time() - current_ts
    influx.write_lag("rps", lag)


async def start_metric_collector(metric_type: str, prom_client, feature_builders, predictors):
    builder = feature_builders[metric_type]
    predictor = predictors[metric_type]

    while True:
        try:
            await _collect_once(metric_type, prom_client, builder, predictor)
        except Exception as e:
            logger.error("[%s] Collector error: %s", metric_type, e)
            metric_states[metric_type].errors_total += 1

        await asyncio.sleep(settings.SCRAPE_INTERVAL_SEC)


async def start_collectors(prom_client, feature_builders, predictors):
    tasks = [
        asyncio.create_task(start_metric_collector(metric_type, prom_client, feature_builders, predictors))
        for metric_type in ["cpu", "rps", "error_rate"]
    ]
    return tasks
