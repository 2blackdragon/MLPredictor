from prometheus_client import Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

# Используем отдельный registry чтобы не смешиваться с дефолтными метриками процесса
registry = CollectorRegistry()

cpu_actual = Gauge(
    "cpu_usage_actual_percent",
    "Текущее значение CPU% из тестового сервиса",
    registry=registry,
)

cpu_predicted = Gauge(
    "cpu_usage_predicted_percent",
    "Предсказанное моделью значение CPU% через 2.5 минуты",
    registry=registry,
)

prediction_error = Gauge(
    "cpu_prediction_absolute_error_percent",
    "Абсолютная ошибка последнего предсказания (факт vs прогноз той же точки)",
    registry=registry,
)

collector_lag_seconds = Gauge(
    "collector_last_update_lag_seconds",
    "Сколько секунд прошло с последнего успешного обновления метрик",
    registry=registry,
)

collector_errors_total = Gauge(
    "collector_errors_total",
    "Суммарное количество ошибок сбора/предсказания с момента старта",
    registry=registry,
)


def metrics_output() -> tuple[bytes, str]:
    """Возвращает (payload, content_type) для endpoint /metrics."""
    return generate_latest(registry), CONTENT_TYPE_LATEST
