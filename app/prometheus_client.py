import time
import logging
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class PrometheusClient:
    """Читает метрики из Prometheus через HTTP API."""

    def __init__(self):
        # Читаем из Prometheus тестового сервиса, не из нашего
        self.base_url = settings.TEST_PROMETHEUS_URL
        self.query = "100 * (1 - avg(rate(node_cpu_seconds_total{mode=\"idle\"}[5m])))"

    async def fetch_recent_values(self, n_points: int) -> list[float] | None:
        """
        Запрашивает последние n_points значений cpu_usage_percent.
        Возвращает список float отсортированный по времени (старые → новые),
        или None если данных недостаточно / Prometheus недоступен.
        """
        step = settings.SCRAPE_INTERVAL_SEC
        end = time.time()
        # Запрашиваем с небольшим запасом по времени
        start = end - (n_points + 5) * step

        url = f"{self.base_url}/api/v1/query_range"
        params = {
            "query": self.query,
            "start": start,
            "end": end,
            "step": step,
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("Prometheus HTTP error: %s", e)
            return None

        data = resp.json()

        if data.get("status") != "success":
            logger.error("Prometheus returned non-success: %s", data)
            return None

        results = data["data"]["result"]
        if not results:
            logger.warning("No results for metric '%s'", self.query)
            return None

        # Берём первый (и обычно единственный) временной ряд
        values: list[float] = [float(v) for _, v in results[0]["values"]]

        if len(values) < n_points:
            logger.warning(
                "Not enough data: got %d points, need %d", len(values), n_points
            )
            return None

        # Возвращаем последние n_points
        return values[-n_points:]

    async def fetch_current_value(self) -> float | None:
        """Текущее (instant) значение метрики."""
        url = f"{self.base_url}/api/v1/query"
        params = {"query": self.query}

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("Prometheus instant query error: %s", e)
            return None

        data = resp.json()
        results = data["data"]["result"]
        if not results:
            return None

        return float(results[0]["value"][1])
