import numpy as np
import pandas as pd
from app.config import MetricConfig


class FeatureBuilder:
    def __init__(self, metric_config: MetricConfig):
        self.metric_config = metric_config
        self.metric_type = metric_config.metric_type
        self.lags = metric_config.lag_list
        self.windows = metric_config.rolling_windows
        self.categorical_features = metric_config.categorical_features

    @property
    def feature_names(self) -> list[str]:
        names = []
        for lag in self.lags:
            names.append(f"{self.metric_type}_lag_{lag}")
        for w in self.windows:
            for stat in ("mean", "std", "min", "max"):
                names.append(f"{self.metric_type}_rolling_{stat}_{w}")
        return names

    def build_features(self, values: list[float], categorical_value: int = None) -> pd.DataFrame:
        if len(values) < max(self.lags):
            raise ValueError(
                f"Need at least {max(self.lags)} points, got {len(values)}"
            )

        arr = np.array(values, dtype=float)
        row: dict[str, float] = {}

        for lag in self.lags:
            row[f"{self.metric_type}_lag_{lag}"] = float(arr[-lag])

        for w in self.windows:
            window_vals = arr[-w:]
            row[f"{self.metric_type}_rolling_mean_{w}"] = float(np.mean(window_vals))
            row[f"{self.metric_type}_rolling_std_{w}"] = float(np.std(window_vals, ddof=1) if len(window_vals) > 1 else 0.0)
            row[f"{self.metric_type}_rolling_min_{w}"] = float(np.min(window_vals))
            row[f"{self.metric_type}_rolling_max_{w}"] = float(np.max(window_vals))

        if categorical_value is not None and self.categorical_features:
            for cat_feature in self.categorical_features:
                row[cat_feature] = categorical_value

        return pd.DataFrame([row], columns=self.metric_config.feature_cols)
