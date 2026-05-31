import requests
import csv
from datetime import datetime, timedelta
import os

PROMETHEUS_URL = "http://138.16.162.15:9090"

QUERY_TEMPLATE = """
sum by (handler) (
  rate(
    http_requests_total{
      handler!="none",
      handler!="/metrics"
    }[5m]
  )
)
"""

START = datetime.fromisoformat("2026-05-25T18:42:00+00:00")
END = datetime.fromisoformat("2026-05-30T00:00:00+00:00")

STEP = "15s"
CHUNK_HOURS = 6

DATA_DIR = "../data"


# =========================
# Prometheus fetch (chunk)
# =========================
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


# =========================
# Chunk loader
# =========================
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


# =========================
# CSV per handler
# =========================
def save_by_handler(results):
    os.makedirs(DATA_DIR, exist_ok=True)

    for series in results:
        handler = series["metric"].get("handler", "unknown")

        safe_name = handler.replace("/", "_").replace("\\", "_")
        if safe_name.startswith("_"):
            safe_name = safe_name[1:]

        filepath = os.path.join(DATA_DIR, f"{safe_name}_rps.csv")

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "timestamp",
                "datetime_utc",
                "handler",
                "value"
            ])

            for ts, value in series["values"]:
                ts = float(ts)

                dt_utc = datetime.utcfromtimestamp(ts)

                writer.writerow([
                    int(ts),
                    dt_utc.isoformat(),
                    handler,
                    float(value)
                ])


# =========================
# RUN
# =========================
query = QUERY_TEMPLATE.replace("[5m]", "[5m]")

results = fetch_all(query)

save_by_handler(results)

print("Done")