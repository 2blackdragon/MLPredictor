import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse

from app.config import settings
from app.prometheus_client import PrometheusClient
from app.feature_builder import FeatureBuilder
from app.predictor import CatBoostPredictor
from app.collector import collector_loop, state
from app.metrics_registry import metrics_output

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация при старте
    prom = PrometheusClient()
    builder = FeatureBuilder()
    predictor = CatBoostPredictor()

    task = asyncio.create_task(collector_loop(prom, builder, predictor))
    logger.info("Background collector task started")

    yield  # Сервис работает

    # Завершение
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    logger.info("Collector task stopped")


app = FastAPI(
    title="CPU Forecast Monitoring",
    description="Сервис мониторинга CPU с ML-прогнозом на 2.5 минуты вперёд",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus scrape endpoint."""
    payload, content_type = metrics_output()
    return Response(content=payload, media_type=content_type)


@app.get("/health")
async def health():
    """Проверка работоспособности сервиса."""
    now = time.time()
    last_ok = state.last_update_ts

    lag = now - last_ok if last_ok else None
    healthy = lag is not None and lag < settings.SCRAPE_INTERVAL_SEC * 3

    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "ok" if healthy else "degraded",
            "last_update_lag_seconds": round(lag, 1) if lag else None,
            "errors_total": state.errors_total,
            "last_actual_cpu": state.last_actual,
            "last_predicted_cpu": state.last_predicted,
            "forecast_horizon_seconds": (
                settings.FORECAST_HORIZON * settings.SCRAPE_INTERVAL_SEC
            ),
        },
    )


@app.get("/debug/features")
async def debug_features():
    """Последний вектор фичей — удобно при отладке."""
    return {
        "feature_count": len(state.last_features),
        "features": state.last_features,
        "last_update_ts": state.last_update_ts,
    }


@app.get("/debug/config")
async def debug_config():
    """Текущий конфиг сервиса."""
    return {
        "prometheus_url": settings.PROMETHEUS_URL,
        "cpu_metric": settings.CPU_METRIC_NAME,
        "scrape_interval_sec": settings.SCRAPE_INTERVAL_SEC,
        "forecast_horizon_steps": settings.FORECAST_HORIZON,
        "forecast_horizon_seconds": (
            settings.FORECAST_HORIZON * settings.SCRAPE_INTERVAL_SEC
        ),
        "history_points": settings.HISTORY_POINTS,
        "lag_list": settings.LAG_LIST,
        "rolling_windows": settings.ROLLING_WINDOWS,
    }
