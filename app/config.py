from pydantic_settings import BaseSettings
import json


class MetricConfig:
    def __init__(
        self,
        metric_type: str,
        prometheus_query: str,
        influx_measurement: str,
        model_path: str,
        model_config_path: str,
        lag_list: list[int] = None,
        rolling_windows: list[int] = None,
    ):
        self.metric_type = metric_type
        self.prometheus_query = prometheus_query
        self.influx_measurement = influx_measurement
        self.model_path = model_path
        self.model_config_path = model_config_path
        self.lag_list = lag_list or [1, 2, 3, 4, 5, 8, 12, 16, 20]
        self.rolling_windows = rolling_windows or [4, 8, 12]
        self._load_feature_cols()

    def _load_feature_cols(self):
        try:
            with open(self.model_config_path, 'r') as f:
                config = json.load(f)
                self.feature_cols = config.get('feature_cols', [])
                self.categorical_features = config.get('categorical_features', [])
        except Exception:
            self.feature_cols = []
            self.categorical_features = []


class Settings(BaseSettings):
    TEST_PROMETHEUS_URL: str = "http://138.16.162.15:9090"

    # InfluxDB — основное хранилище метрик
    INFLUXDB_URL: str = "http://influxdb:8086"
    INFLUXDB_TOKEN: str = "my-super-secret-token"
    INFLUXDB_ORG: str = "mlmonitor"
    INFLUXDB_BUCKET: str = "cpu_monitor"

    # Расписание сбора (должно совпадать с шагом при обучении)
    SCRAPE_INTERVAL_SEC: int = 15

    # Параметры модели
    FORECAST_HORIZON: int = 10          # шагов × 15с = 2.5 мин
    MAX_LAG: int = 20
    LAG_LIST: list[int] = [1, 2, 3, 4, 5, 8, 12, 16, 20]
    ROLLING_WINDOWS: list[int] = [4, 8, 12]

    # Сколько точек запрашивать из Prometheus (с запасом)
    HISTORY_POINTS: int = 30            # 30 × 15с = 7.5 мин истории

    class Config:
        env_file = ".env"

    def get_metric_config(self, metric_type: str) -> MetricConfig:
        configs = {
            "cpu": MetricConfig(
                metric_type="cpu",
                prometheus_query="100 * (1 - avg(rate(node_cpu_seconds_total{mode=\"idle\"}[5m])))",
                influx_measurement="cpu_usage",
                model_path="models/cpu_forecast_model.cbm",
                model_config_path="models/cpu_model_config.json",
            ),
            "rps": MetricConfig(
                metric_type="rps",
                prometheus_query='sum by (handler) (rate(http_requests_total{handler!="none",handler!="/metrics"}[5m]))',
                influx_measurement="rps",
                model_path="models/rps_forecast_model.cbm",
                model_config_path="models/rps_model_config.json",
            ),
            "error_rate": MetricConfig(
                metric_type="error_rate",
                prometheus_query="sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m]))",
                influx_measurement="error_rate",
                model_path="models/error_rate_forecast_model.cbm",
                model_config_path="models/error_rate_model_config.json",
            ),
        }
        return configs.get(metric_type)


settings = Settings()
