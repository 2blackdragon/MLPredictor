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

    def write_cpu_metrics(self, actual: float, predicted: float, horizon_seconds: int):
        """Записываем actual и predicted с будущим timestamp для predicted"""
        now = datetime.now(timezone.utc)

        # Actual — текущее время
        point_actual = Point("cpu_usage") \
            .tag("type", "actual") \
            .field("percent", actual) \
            .time(now, WritePrecision.S)

        # Predicted — с будущим временем
        future_time = now.timestamp() + horizon_seconds
        point_predicted = Point("cpu_usage") \
            .tag("type", "predicted") \
            .tag("horizon", f"{horizon_seconds}s") \
            .field("percent", predicted) \
            .time(int(future_time * 1000), WritePrecision.MS)  # миллисекунды

        try:
            self.write_api.write(bucket=self.bucket, record=[point_actual, point_predicted])
        except Exception as e:
            logger.error("Failed to write CPU metrics to InfluxDB: %s", e)
            raise

    def write_error(self, error: float):
        point = Point("prediction_error") \
            .field("absolute_percent", error) \
            .time(datetime.now(timezone.utc), WritePrecision.S)
        try:
            self.write_api.write(bucket=self.bucket, record=point)
        except Exception as e:
            logger.error("Failed to write error metric to InfluxDB: %s", e)

    def write_lag(self, lag_seconds: float):
        point = Point("collector") \
            .field("lag_seconds", lag_seconds) \
            .time(datetime.now(timezone.utc), WritePrecision.S)
        try:
            self.write_api.write(bucket=self.bucket, record=point)
        except Exception as e:
            logger.error("Failed to write lag metric to InfluxDB: %s", e)

    def close(self):
        self.client.close()
        logger.info("InfluxDB client closed")
