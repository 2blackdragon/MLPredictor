from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.config import MetricConfig


class RidgeFeatureBuilder:
    """
    Ridge feature builder mirroring `make_features` from the walk-forward
    training notebooks (Yandex Handbook §10, Scheme 2). Used for all metrics
    (CPU / RPS / error_rate).

    For each new point `y_t` we treat it as the "current" observation; the
    notebook used `series.shift(1).rolling(...)` so rolling stats / EWM are
    computed on history *excluding* the current point. Lags are aligned to
    the same convention (`lag_k` = value at `t - k`).
    """

    def __init__(self, metric_config: MetricConfig):
        self.metric_config = metric_config
        self.metric_type = metric_config.metric_type
        self.lags = metric_config.lag_list
        self.windows = metric_config.rolling_windows
        self.ewm_spans = metric_config.ewm_spans or []
        self._declared_cols = metric_config.feature_cols

    @property
    def feature_names(self) -> list[str]:
        if self._declared_cols:
            return list(self._declared_cols)
        names = [f"lag_{l}" for l in self.lags]
        names += [f"roll_mean_{w}" for w in self.windows]
        names += [f"roll_std_{w}" for w in self.windows]
        names += [f"ewm_{s}" for s in self.ewm_spans]
        names += ["hour_sin", "hour_cos", "minute_sin", "minute_cos", "diff_1", "diff_20"]
        return names

    @property
    def min_history_points(self) -> int:
        # Largest lag/window referenced + 1 (we need values strictly before the
        # current point for shift(1)-style rolling stats).
        biggest = max(self.lags + self.windows + [20])
        return biggest + 1

    def build_features(
        self,
        values: list[float],
        categorical_value: int | None = None,  # noqa: ARG002 — unused for Ridge
        current_ts: float | None = None,
    ) -> pd.DataFrame:
        need = self.min_history_points
        if len(values) < need:
            raise ValueError(
                f"Ridge feature builder needs at least {need} points, got {len(values)}"
            )

        arr = np.asarray(values, dtype=float)
        # "Current" point = last value; history used for rolling/ewm is arr[:-1]
        history = arr[:-1]

        row: dict[str, float] = {}

        # Lags: lag_k = value at t-k. arr[-1] is t, arr[-1-k] is t-k.
        for lag in self.lags:
            row[f"lag_{lag}"] = float(arr[-1 - lag])

        # Rolling mean / std over the last `w` history points (shift(1) semantics).
        for w in self.windows:
            window_vals = history[-w:]
            row[f"roll_mean_{w}"] = float(np.mean(window_vals))
            row[f"roll_std_{w}"] = float(
                np.std(window_vals, ddof=1) if len(window_vals) > 1 else 0.0
            )

        # Exponentially weighted mean over history (shift(1).ewm(span=s).mean())
        for span in self.ewm_spans:
            row[f"ewm_{span}"] = float(_ewm_last(history, span))

        # Time-of-day features taken from the timestamp of the *current* point.
        if current_ts is None:
            current_ts = datetime.now(timezone.utc).timestamp()
        dt = datetime.fromtimestamp(current_ts, tz=timezone.utc)
        hour, minute = dt.hour, dt.minute
        row["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
        row["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
        row["minute_sin"] = float(np.sin(2 * np.pi * minute / 60))
        row["minute_cos"] = float(np.cos(2 * np.pi * minute / 60))

        # diff_k = series.diff(k).shift(1) — i.e. (y_{t-1} - y_{t-1-k})
        row["diff_1"] = float(arr[-2] - arr[-3]) if len(arr) >= 3 else 0.0
        row["diff_20"] = (
            float(arr[-2] - arr[-22]) if len(arr) >= 22 else 0.0
        )

        # Silence pandas/numpy warnings about NaNs from too-short windows.
        for k, v in list(row.items()):
            if not np.isfinite(v):
                row[k] = 0.0

        cols = self._declared_cols or self.feature_names
        return pd.DataFrame([row], columns=cols)


def _ewm_last(values: np.ndarray, span: int) -> float:
    """
    Last value of an exponentially weighted mean with the same semantics as
    `pandas.Series.ewm(span=s, adjust=True).mean()`.
    """
    if len(values) == 0:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    # adjust=True formula: sum(w_i * x_i) / sum(w_i),  w_i = (1-alpha)^i  (i from newest=0)
    n = len(values)
    weights = (1 - alpha) ** np.arange(n)[::-1]  # oldest gets highest power
    return float(np.sum(weights * values) / np.sum(weights))


# Backwards-compatible alias for collector type-hints.
FeatureBuilder = RidgeFeatureBuilder


def build_feature_builder(metric_config: MetricConfig) -> RidgeFeatureBuilder:
    return RidgeFeatureBuilder(metric_config)
