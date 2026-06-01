import time
import logging
import httpx
from app.config import MetricConfig

logger = logging.getLogger(__name__)


class PrometheusClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def fetch_recent_values(
        self,
        query: str,
        n_points: int,
        scrape_interval: int,
    ) -> list[float] | None:
        step = scrape_interval
        end = time.time()
        start = end - (n_points + 5) * step

        url = f"{self.base_url}/api/v1/query_range"
        params = {
            "query": query,
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
            logger.warning("No results for metric")
            return None

        values: list[float] = [float(v) for _, v in results[0]["values"]]

        if len(values) < n_points:
            logger.warning(
                "Not enough data: got %d points, need %d", len(values), n_points
            )
            return None

        return values[-n_points:]

    async def fetch_recent_values_per_series(
        self,
        query: str,
        n_points: int,
        scrape_interval: int,
        label_name: str = "handler",
    ) -> dict[str, list[float]] | None:
        step = scrape_interval
        end = time.time()
        start = end - (n_points + 5) * step

        url = f"{self.base_url}/api/v1/query_range"
        params = {
            "query": query,
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
            logger.warning("No results for metric")
            return None

        result_dict: dict[str, list[float]] = {}

        for result in results:
            metric = result.get("metric", {})
            label_value = metric.get(label_name)

            if not label_value:
                logger.warning("Series missing '%s' label: %s", label_name, metric)
                continue

            values: list[float] = [float(v) for _, v in result["values"]]

            if len(values) < n_points:
                logger.warning(
                    "Not enough data for %s=%s: got %d points, need %d",
                    label_name,
                    label_value,
                    len(values),
                    n_points,
                )
                continue

            result_dict[label_value] = values[-n_points:]

        if not result_dict:
            logger.warning("No valid series found for metric")
            return None

        return result_dict

    async def fetch_current_value(self, query: str) -> float | None:
        url = f"{self.base_url}/api/v1/query"
        params = {"query": query}

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
