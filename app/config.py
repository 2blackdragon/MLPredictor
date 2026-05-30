from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Prometheus тестового сервиса — источник cpu_usage_percent
    # Адрес резолвится через внешнюю docker network (test-network)
    TEST_PROMETHEUS_URL: str = "http://138.16.162.15:9090"
    CPU_METRIC_NAME: str = "cpu_usage_percent"

    # Наш Prometheus — только для хранения предсказаний и экспозиции в Grafana
    PROMETHEUS_URL: str = "http://prometheus:9090"

    # Расписание сбора (должно совпадать с шагом при обучении)
    SCRAPE_INTERVAL_SEC: int = 15

    # Параметры модели (из model_config_stratified_clean.json)
    FORECAST_HORIZON: int = 10          # шагов × 15с = 2.5 мин
    MAX_LAG: int = 20
    LAG_LIST: list[int] = [1, 2, 3, 4, 5, 8, 12, 16, 20]
    ROLLING_WINDOWS: list[int] = [4, 8, 12]

    # Сколько точек запрашивать из Prometheus (с запасом)
    HISTORY_POINTS: int = 30            # 30 × 15с = 7.5 мин истории

    # Пути к модели
    MODEL_PATH: str = "model/cpu_forecast_model_stratified_clean.cbm"
    MODEL_CONFIG_PATH: str = "model/model_config_stratified_clean.json"

    class Config:
        env_file = ".env"


settings = Settings()
