import logging
from datetime import datetime, timezone
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from app.config import settings

logger = logging.getLogger(__name__)


class InfluxWriter:
    def __init__(self):
        self.url = settings.INFLUXDB_URL
        self.token = settings.INFLUXDB_TOKEN
        self.org = settings.INFLUXDB_ORG
        self.bucket = settings.INFLUXDB_BUCKET

        try:
            self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            logger.info("InfluxDB client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize InfluxDB client: %s", e)
            raise

    def write_metric(
        self,
        metric_type: str,
        measurement: str,
        actual: float,
        predicted: float,
        horizon_seconds: int,
    ):
        now = datetime.now(timezone.utc)

        point_actual = Point(measurement).tag("type", "actual").field("value", actual).time(now, WritePrecision.S)

        future_time = now.timestamp() + horizon_seconds
        point_predicted = (
            Point(measurement)
            .tag("type", "predicted")
            .tag("horizon", f"{horizon_seconds}s")
            .field("value", predicted)
            .time(int(future_time * 1000), WritePrecision.MS)
        )

        try:
            self.write_api.write(self.bucket, record=[point_actual, point_predicted])
        except Exception as e:
            logger.error("Failed to write metrics for '%s' to InfluxDB: %s", metric_type, e)
            raise

    def write_error(self, metric_type: str, error: float):
        point = (
            Point(f"{metric_type}_error")
            .field("absolute", error)
            .time(datetime.now(timezone.utc), WritePrecision.S)
        )
        try:
            self.write_api.write(self.bucket, record=point)
        except Exception as e:
            logger.error("Failed to write error metric for '%s' to InfluxDB: %s", metric_type, e)

    def write_lag(self, metric_type: str, lag_seconds: float):
        point = (
            Point(f"{metric_type}_collector")
            .field("lag_seconds", lag_seconds)
            .time(datetime.now(timezone.utc), WritePrecision.S)
        )
        try:
            self.write_api.write(self.bucket, record=point)
        except Exception as e:
            logger.error("Failed to write lag metric for '%s' to InfluxDB: %s", metric_type, e)

    def write_ml_metrics(self, metric_type: str, r2: float, mae: float, rmse: float):
        point = (
            Point(f"{metric_type}_ml_quality")
            .field("r2", r2)
            .field("mae", mae)
            .field("rmse", rmse)
            .time(datetime.now(timezone.utc), WritePrecision.S)
        )
        try:
            self.write_api.write(self.bucket, record=point)
        except Exception as e:
            logger.error("Failed to write ML quality metrics for '%s' to InfluxDB: %s", metric_type, e)

    def close(self):
        self.client.close()
        logger.info("InfluxDB client closed")
