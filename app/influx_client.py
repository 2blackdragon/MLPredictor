import logging
from datetime import datetime, timezone

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from app.config import settings

logger = logging.getLogger(__name__)


class InfluxWriter:
    """
    Unified InfluxDB writer:
    - consistent schema for all metrics
    - Grafana-friendly structure
    - single field: value
    - tags: type, handler, horizon
    """

    def __init__(self):
        self.url = settings.INFLUXDB_URL
        self.token = settings.INFLUXDB_TOKEN
        self.org = settings.INFLUXDB_ORG
        self.bucket = settings.INFLUXDB_BUCKET

        try:
            self.client = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org,
            )
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            logger.info("InfluxDB client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize InfluxDB client: %s", e)
            raise

    # -----------------------------
    # MAIN METRIC WRITER
    # -----------------------------
    def write_metric(
        self,
        metric_type: str,
        measurement: str,
        actual: float,
        predicted: float,
        horizon_seconds: int,
        handler: str | None = None,
    ):
        """
        Writes:
        - actual value (current timestamp)
        - predicted value (future timestamp)
        """

        now = datetime.now(timezone.utc)

        # -----------------
        # ACTUAL
        # -----------------
        point_actual = (
            Point(measurement)
            .tag("metric", metric_type)
            .tag("type", "actual")
            .field("value", float(actual))
            .time(now, WritePrecision.S)
        )

        if handler:
            point_actual = point_actual.tag("handler", handler)

        # -----------------
        # PREDICTED
        # -----------------
        future_ts = now.timestamp() + horizon_seconds

        point_predicted = (
            Point(measurement)
            .tag("metric", metric_type)
            .tag("type", "predicted")
            .tag("horizon", f"{horizon_seconds}s")
            .field("value", float(predicted))
            .time(int(future_ts * 1000), WritePrecision.MS)
        )

        if handler:
            point_predicted = point_predicted.tag("handler", handler)

        try:
            self.write_api.write(
                bucket=self.bucket,
                record=[point_actual, point_predicted],
            )
        except Exception as e:
            logger.error("Failed to write metric '%s': %s", metric_type, e)
            raise

    # -----------------------------
    # ERROR METRICS
    # -----------------------------
    def write_error(
        self,
        metric_type: str,
        error: float,
        handler: str | None = None,
    ):
        """
        Unified error format:
        measurement = {metric}_error
        field = value
        """

        point = (
            Point(f"{metric_type}_error")
            .tag("type", "actual")
            .field("value", float(error))
            .time(datetime.now(timezone.utc), WritePrecision.S)
        )

        if handler:
            point = point.tag("handler", handler)

        try:
            self.write_api.write(self.bucket, record=point)
        except Exception as e:
            logger.error("Failed to write error metric '%s': %s", metric_type, e)

    # -----------------------------
    # COLLECTOR LAG
    # -----------------------------
    def write_lag(
        self,
        metric_type: str,
        lag_seconds: float,
    ):
        point = (
            Point("collector_lag")
            .tag("metric", metric_type)
            .field("value", float(lag_seconds))
            .time(datetime.now(timezone.utc), WritePrecision.S)
        )

        try:
            self.write_api.write(self.bucket, record=point)
        except Exception as e:
            logger.error("Failed to write lag metric '%s': %s", metric_type, e)

    # -----------------------------
    # ML METRICS
    # -----------------------------
    def write_ml_metrics(
        self,
        metric_type: str,
        r2: float,
        mae: float,
        rmse: float,
        handler: str | None = None,
    ):
        point = (
            Point(f"{metric_type}_ml_quality")
            .tag("metric", metric_type)
            .field("r2", float(r2))
            .field("mae", float(mae))
            .field("rmse", float(rmse))
            .time(datetime.now(timezone.utc), WritePrecision.S)
        )

        if handler:
            point = point.tag("handler", handler)

        try:
            self.write_api.write(self.bucket, record=point)
        except Exception as e:
            logger.error("Failed to write ML metrics '%s': %s", metric_type, e)

    # -----------------------------
    # CLOSE
    # -----------------------------
    def close(self):
        try:
            self.client.close()
            logger.info("InfluxDB client closed")
        except Exception as e:
            logger.error("Error closing InfluxDB client: %s", e)
