import requests
import csv
from datetime import datetime, timedelta, timezone

PROMETHEUS_URL = "http://138.16.162.15:9090"

START = datetime.fromisoformat("2026-05-25T18:42:00+00:00")
END = datetime.fromisoformat("2026-05-31T00:15:20+00:00")

STEP = "15s"

ERROR_RATE_QUERY = """
sum(rate(http_requests_total{status=~"4..|5.."}[5m])) / ignoring(status) sum(rate(http_requests_total[5m]))
"""

CHUNK_HOURS = 6


def fetch_range(query, start_dt, end_dt):
    url = f"{PROMETHEUS_URL}/api/v1/query_range"

    resp = requests.get(url, params={
        "query": query,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "step": STEP
    })

    resp.raise_for_status()

    data = resp.json()

    if data["status"] != "success":
        raise Exception(data)

    return data["data"]["result"]


def fetch_all(query):
    current = START
    merged = []

    while current < END:
        chunk_end = min(current + timedelta(hours=CHUNK_HOURS), END)

        print(f"Fetching {current} -> {chunk_end}")

        results = fetch_range(query, current, chunk_end)
        merged.extend(results)

        current = chunk_end

    return merged


def save_csv(filename, results, value_name):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            value_name
        ])

        for series in results:
            for ts, value in series["values"]:
                try:
                    writer.writerow([
                        int(float(ts)),
                        float(value)
                    ])
                except ValueError:
                    # пропускаем NaN / пустые значения (деление на 0 когда нет запросов)
                    pass


error_rate_results = fetch_all(ERROR_RATE_QUERY)
save_csv("../data/error_rate_dataset.csv", error_rate_results, "error_rate")

print("Done")