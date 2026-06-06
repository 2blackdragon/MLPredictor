import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import settings
from app.prometheus_client import PrometheusClient
from app.feature_builder import build_feature_builder
from app.predictor import build_predictor
from app.collector import start_collectors, metric_states

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    prom_client = PrometheusClient(settings.TEST_PROMETHEUS_URL)

    feature_builders = {}
    predictors = {}

    for metric_type in ["cpu", "rps", "error_rate"]:
        metric_config = settings.get_metric_config(metric_type)
        feature_builders[metric_type] = build_feature_builder(metric_config)
        predictors[metric_type] = build_predictor(metric_config)
        logger.info(
            "Initialized %s metric (model_type=%s, horizon=%ds, history=%d pts)",
            metric_type,
            metric_config.model_type,
            metric_config.forecast_horizon_seconds,
            metric_config.min_history_points,
        )

    tasks = await start_collectors(prom_client, feature_builders, predictors)
    logger.info("All background collector tasks started")

    yield

    for task in tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    logger.info("All collector tasks stopped")


app = FastAPI(
    title="Multi-Metric Forecast Monitoring",
    description="Сервис мониторинга CPU, RPS, Error Rate с ML-прогнозом на 2.5 минуты вперёд",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    results = {}
    for metric_type in ["cpu", "rps", "error_rate"]:
        state = metric_states[metric_type]
        now = time.time()
        last_ok = state.last_update_ts

        lag = now - last_ok if last_ok else None
        healthy = lag is not None and lag < settings.SCRAPE_INTERVAL_SEC * 3

        results[metric_type] = {
            "status": "ok" if healthy else "degraded",
            "last_update_lag_seconds": round(lag, 1) if lag else None,
            "errors_total": state.errors_total,
            "last_actual": state.last_actual,
            "last_predicted": state.last_predicted,
            "forecast_horizon_seconds": settings.get_metric_config(
                metric_type
            ).forecast_horizon_seconds,
        }

    overall_healthy = all(r["status"] == "ok" for r in results.values())

    return JSONResponse(
        status_code=200 if overall_healthy else 503,
        content={
            "overall_status": "ok" if overall_healthy else "degraded",
            "metrics": results,
        },
    )


@app.get("/debug/features/{metric_type}")
async def debug_features(metric_type: str):
    if metric_type not in metric_states:
        return {"error": f"Unknown metric type: {metric_type}"}, 404

    state = metric_states[metric_type]
    return {
        "metric_type": metric_type,
        "feature_count": len(state.last_features),
        "features": state.last_features,
        "last_update_ts": state.last_update_ts,
    }


@app.get("/debug/config")
async def debug_config():
    return {
        "prometheus_url": settings.TEST_PROMETHEUS_URL,
        "scrape_interval_sec": settings.SCRAPE_INTERVAL_SEC,
        "forecast_horizon_steps": settings.FORECAST_HORIZON,
        "forecast_horizon_seconds": (
            settings.FORECAST_HORIZON * settings.SCRAPE_INTERVAL_SEC
        ),
        "history_points": settings.HISTORY_POINTS,
        "lag_list": settings.LAG_LIST,
        "rolling_windows": settings.ROLLING_WINDOWS,
        "storage_backend": "influxdb",
        "influxdb_url": settings.INFLUXDB_URL,
        "influxdb_org": settings.INFLUXDB_ORG,
        "influxdb_bucket": settings.INFLUXDB_BUCKET,
    }
