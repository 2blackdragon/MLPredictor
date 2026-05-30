import numpy as np
import pandas as pd

from app.config import settings


class FeatureBuilder:
    """
    Воспроизводит create_features() из ноутбука.

    Принимает список float (последние HISTORY_POINTS значений CPU,
    отсортированных старые→новые), возвращает DataFrame из одной строки
    с теми же колонками что при обучении модели.
    """

    def __init__(self):
        self.lags = settings.LAG_LIST
        self.windows = settings.ROLLING_WINDOWS

    @property
    def feature_names(self) -> list[str]:
        names = []
        for lag in self.lags:
            names.append(f"cpu_lag_{lag}")
        for w in self.windows:
            for stat in ("mean", "std", "min", "max"):
                names.append(f"cpu_rolling_{stat}_{w}")
        return names

    def build(self, values: list[float]) -> pd.DataFrame:
        """
        values — список float, последнее значение = values[-1] = текущий CPU.
        Минимальная длина: max(lags) = 20 точек.
        """
        if len(values) < max(self.lags):
            raise ValueError(
                f"Need at least {max(self.lags)} points, got {len(values)}"
            )

        arr = np.array(values, dtype=float)
        row: dict[str, float] = {}

        # Lag features: lag_1 = values[-1] (1 шаг назад), lag_2 = values[-2], ...
        # При обучении: df[f'cpu_lag_{lag}'] = df['cpu_usage_percent'].shift(lag)
        # Для текущей строки shift(lag) означает значение lag шагов назад
        for lag in self.lags:
            row[f"cpu_lag_{lag}"] = float(arr[-lag])

        # Rolling features: вычисляем по последним window точкам
        # При обучении rolling(window) на текущей строке = среднее по [t-window+1 .. t]
        for w in self.windows:
            window_vals = arr[-w:]
            row[f"cpu_rolling_mean_{w}"] = float(np.mean(window_vals))
            row[f"cpu_rolling_std_{w}"] = float(np.std(window_vals, ddof=1) if len(window_vals) > 1 else 0.0)
            row[f"cpu_rolling_min_{w}"] = float(np.min(window_vals))
            row[f"cpu_rolling_max_{w}"] = float(np.max(window_vals))

        return pd.DataFrame([row], columns=self.feature_names)
