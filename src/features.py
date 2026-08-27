"""
features.py
Shared feature engineering for the weather model.

Core idea: to predict "today's" weather, we can only honestly use
information known BEFORE today ends -- i.e. yesterday's and recent
prior days' observed weather, plus calendar/seasonal signals.
"""

import numpy as np
import pandas as pd

LAGS = [1, 2, 3]  # days back to use as features


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: df with columns [date, temperature_2m_max, temperature_2m_min,
    precipitation_sum, windspeed_10m_max, relative_humidity_2m_mean,
    surface_pressure_mean], sorted or not, one row per day.

    Output: df with lag features + seasonal features + targets for "today":
      - target_temp_max   (today's actual max temp -- regression target)
      - target_rain       (1 if today's precipitation > 0 -- classification target)
    """
    df = df.sort_values("date").reset_index(drop=True)

    base_cols = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "windspeed_10m_max",
        "relative_humidity_2m_mean",
        "surface_pressure_mean",
    ]

    feat = pd.DataFrame({"date": df["date"]})

    for col in base_cols:
        for lag in LAGS:
            feat[f"{col}_lag{lag}"] = df[col].shift(lag)

    # simple trend features: change between yesterday and 2 days ago
    feat["temp_max_trend"] = df["temperature_2m_max"].shift(1) - df["temperature_2m_max"].shift(2)
    feat["pressure_trend"] = df["surface_pressure_mean"].shift(1) - df["surface_pressure_mean"].shift(2)

    # seasonal signal (day of year, cyclically encoded)
    doy = df["date"].dt.dayofyear
    feat["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feat["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # targets = today's actual observed values
    feat["target_temp_max"] = df["temperature_2m_max"]
    feat["target_rain"] = (df["precipitation_sum"] > 0).astype(int)

    # drop rows without enough lag history
    feat = feat.dropna().reset_index(drop=True)
    return feat


def feature_columns(feat: pd.DataFrame):
    return [c for c in feat.columns if c not in ("date", "target_temp_max", "target_rain")]
