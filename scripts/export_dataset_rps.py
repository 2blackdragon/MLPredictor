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
END = datetime.fromisoformat("2026-05-31T00:15:20+00:00")

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

    # 1. Группируем данные по хендлерам и дедуплицируем по таймстампу
    # Используем словарь {ts: value}, чтобы автоматически убрать дубли на стыках чанков
    handler_data = {}
    
    for series in results:
        handler = series["metric"].get("handler", "unknown")
        if handler not in handler_data:
            handler_data[handler] = {}
            
        for ts, value in series["values"]:
            handler_data[handler][ts] = value

    # 2. Записываем сгруппированные данные (по одному открытию файла на хендлер)
    for handler, ts_dict in handler_data.items():
        safe_name = handler.replace("/", "_").replace("\\", "_")
        if safe_name.startswith("_"):
            safe_name = safe_name[1:]

        filepath = os.path.join(DATA_DIR, f"{safe_name}_rps.csv")

        # Сортируем таймстампы по возрастанию, чтобы данные шли хронологически
        sorted_timestamps = sorted(ts_dict.keys(), key=float)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            # Записываем заголовок один раз для всего файла
            writer.writerow([
                "timestamp",
                "datetime_utc",
                "handler",
                "value"
            ])

            for ts in sorted_timestamps:
                value = ts_dict[ts]
                ts_float = float(ts)
                dt_utc = datetime.utcfromtimestamp(ts_float)

                writer.writerow([
                    int(ts_float),
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