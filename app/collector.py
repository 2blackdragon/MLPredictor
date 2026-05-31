import asyncio
import time
from collections import deque
import logging
from app.influx_client import InfluxWriter
from app.feature_builder import FeatureBuilder
from app.predictor import CatBoostPredictor
from app.config import settings

logger = logging.getLogger(__name__)


import math  # Добавьте импорт в начало файла

class CollectorState:
    def __init__(self):
        self.predictions_buffer: deque = deque(maxlen=500)
        
        # --- НОВЫЕ БУФЕРЫ ДЛЯ РАСЧЕТА МЕТРИК КАЧЕСТВА ---
        # Храним последние 100 созревших точек для оценки скользящего качества
        self.actual_history: deque = deque(maxlen=100)
        self.predicted_history: deque = deque(maxlen=100)
        
        self.errors_total = 0
        self.last_update_ts: float | None = None
        self.last_actual: float | None = None
        self.last_predicted: float | None = None
        self.last_features: list[float] = []

def calculate_ml_metrics(self) -> dict[str, float] | None:
        n = len(self.actual_history)
        # Нам нужно накопить хотя бы 15-20 точек для адекватного R2
        if n < 15: 
            return None
            
        y_true = list(self.actual_history)
        y_pred = list(self.predicted_history)
        
        # 1. MAE
        mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
        
        # 2. RMSE
        mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n
        rmse = math.sqrt(mse)
        
        # 3. R^2 Score с защитой
        mean_true = sum(y_true) / n
        ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
        ss_tot = sum((t - mean_true) ** 2 for t in y_true)
        
        # Если дисперсии факта почти нет (平), R2 не имеет математического смысла. Установим 1.0, если ошибок нет, или 0.0
        if ss_tot < 1e-5:
            r2 = 1.0 if ss_res < 1e-5 else 0.0
        else:
            r2 = 1.0 - (ss_res / ss_tot)
            
        # Ограничиваем снизу здравым смыслом (например, -1.0), чтобы не ломать графики гигантскими минусами
        r2 = max(r2, -1.0)
        
        return {"r2": r2, "mae": mae, "rmse": rmse}


state = CollectorState()
influx = InfluxWriter()


async def _collect_once(
    prom_client,
    builder: FeatureBuilder,
    predictor: CatBoostPredictor,
) -> None:
    # 1. Получаем актуальные данные
    values = await prom_client.fetch_recent_values(settings.HISTORY_POINTS)
    if values is None or len(values) < settings.HISTORY_POINTS:
        logger.warning("Not enough data")
        state.errors_total += 1
        return

    actual = values[-1]
    current_ts = time.time()

    # 2. Предсказание
    try:
        features_df = builder.build_features(values)
        features_list = features_df.iloc[0].tolist()
        predicted = predictor.predict(features_df)
    except Exception as e:
        logger.error("Prediction error: %s", e)
        state.errors_total += 1
        return

    horizon_seconds = settings.FORECAST_HORIZON * settings.SCRAPE_INTERVAL_SEC

    # 3. Записываем в InfluxDB (predicted с будущим временем!)
    influx.write_cpu_metrics(actual, predicted, horizon_seconds)

    # 4. Оценка ошибки
    # 4. Оценка ошибки и расчет метрик качества
    if len(state.predictions_buffer) >= settings.FORECAST_HORIZON:
        old_predicted = state.predictions_buffer.popleft()
        error = abs(actual - old_predicted)
        influx.write_error(error)
        
        # --- НОВЫЙ БЛОК РАСЧЕТА МЕТРИК МcontentОДЕЛИ ---
        # Добавляем созревшую пару (факт и его прогноз) в историю качества
        state.actual_history.append(actual)
        state.predicted_history.append(old_predicted)
        
        # Считаем метрики
        metrics = state.calculate_ml_metrics()
        if metrics:
            # Отправляем пачкой в InfluxDB (метод write_ml_metrics напишем ниже)
            influx.write_ml_metrics(
                r2=metrics["r2"], 
                mae=metrics["mae"], 
                rmse=metrics["rmse"]
            )
            logger.info(
                f"Metrics updated (window={len(state.actual_history)}) | "
                f"R²={metrics['r2']:.3f} | MAE={metrics['mae']:.2f}% | RMSE={metrics['rmse']:.2f}%"
            )
        # ---------------------------------------------
        
        logger.info(f"Prediction matured | error={error:.3f}%")

    # Добавляем текущее предсказание в буфер
    state.predictions_buffer.append(predicted)

    # Лаг
    lag = time.time() - current_ts
    influx.write_lag(lag)

    # Обновляем состояние для health endpoint
    state.last_update_ts = current_ts
    state.last_actual = actual
    state.last_predicted = predicted
    state.last_features = features_list

    logger.info(
        f"actual={actual:.2f}% | predicted={predicted:.2f}% (+{horizon_seconds}s)"
    )


async def start_collector(prom_client, builder, predictor):
    while True:
        try:
            await _collect_once(prom_client, builder, predictor)
        except Exception as e:
            logger.error("Collector error: %s", e)
            state.errors_total += 1

        await asyncio.sleep(settings.SCRAPE_INTERVAL_SEC)
