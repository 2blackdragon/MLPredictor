from prometheus_client import Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

# Используем отдельный registry
registry = CollectorRegistry()

# ==================== Основные метрики ====================

cpu_actual = Gauge(
    "cpu_usage_actual_percent",
    "Текущее реальное значение CPU% из тестового сервиса",
    registry=registry,
)

# Изменено: теперь с label'ом horizon
cpu_predicted = Gauge(
    "cpu_usage_predicted_percent",
    "Предсказанное значение CPU% на заданный горизонт",
    ["horizon"],          # например: 150s, 300s и т.д.
    registry=registry,
)

prediction_error = Gauge(
    "cpu_prediction_absolute_error_percent",
    "Абсолютная ошибка последнего предсказания (факт vs прогноз)",
    registry=registry,
)

collector_lag_seconds = Gauge(
    "collector_last_update_lag_seconds",
    "Сколько секунд прошло с последнего успешного обновления метрик",
    registry=registry,
)

collector_errors_total = Gauge(
    "collector_errors_total",
    "Суммарное количество ошибок с момента старта",
    registry=registry,
)


def metrics_output() -> tuple[bytes, str]:
    """Возвращает (payload, content_type) для endpoint /metrics."""
    return generate_latest(registry), CONTENT_TYPE_LATEST
