"""
predict_today.py
Fetches the last ~10 days of weather (needed for lag features), then
predicts TODAY's max temperature and rain probability using the
trained models.

Run this daily (see .github/workflows/daily_predict.yml) to get a
fresh prediction every morning.
"""

import argparse
import datetime as dt
import joblib
import pandas as pd

from fetch_data import fetch_history
from features import build_features, feature_columns


def main():
    parser = argparse.ArgumentParser(description="Predict today's weather")
    parser.add_argument("--lat", type=float, default=48.8566, help="Latitude (default: Paris)")
    parser.add_argument("--lon", type=float, default=2.3522, help="Longitude (default: Paris)")
    parser.add_argument("--models-dir", type=str, default="models")
    args = parser.parse_args()

    today = dt.date.today()
    # Fetch a short recent window so we have enough lag history.
    # end date = yesterday, since today's own data doesn't exist yet.
    end = today - dt.timedelta(days=1)
    start = end - dt.timedelta(days=14)

    raw = fetch_history(args.lat, args.lon, start.isoformat(), end.isoformat())

    # Append a placeholder "today" row (values unknown / NaN) so that
    # build_features() produces a feature row whose lag1 = yesterday.
    placeholder = pd.DataFrame([{
        "date": pd.Timestamp(today),
        "temperature_2m_max": None,
        "temperature_2m_min": None,
        "precipitation_sum": None,
        "windspeed_10m_max": None,
        "relative_humidity_2m_mean": None,
        "surface_pressure_mean": None,
    }])
    combined = pd.concat([raw, placeholder], ignore_index=True)

    feat = build_features(combined)
    # after dropna() the placeholder row (which has NaN targets) will be
    # dropped too, so instead we build features manually for the last row
    # using the same helper logic, ignoring the target columns.
    # Simplest robust approach: build features WITHOUT the dropna target filter.
    from features import LAGS
    import numpy as np

    df_sorted = combined.sort_values("date").reset_index(drop=True)
    base_cols = [
        "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
        "windspeed_10m_max", "relative_humidity_2m_mean", "surface_pressure_mean",
    ]
    row = {}
    for col in base_cols:
        for lag in LAGS:
            row[f"{col}_lag{lag}"] = df_sorted[col].shift(lag).iloc[-1]
    row["temp_max_trend"] = df_sorted["temperature_2m_max"].shift(1).iloc[-1] - df_sorted["temperature_2m_max"].shift(2).iloc[-1]
    row["pressure_trend"] = df_sorted["surface_pressure_mean"].shift(1).iloc[-1] - df_sorted["surface_pressure_mean"].shift(2).iloc[-1]
    doy = pd.Timestamp(today).dayofyear
    row["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    row["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    X_today = pd.DataFrame([row])

    temp_bundle = joblib.load(f"{args.models_dir}/temp_model.joblib")
    rain_bundle = joblib.load(f"{args.models_dir}/rain_model.joblib")

    cols = temp_bundle["features"]
    X_today = X_today[cols]

    pred_temp = temp_bundle["model"].predict(X_today)[0]
    pred_rain_prob = rain_bundle["model"].predict_proba(X_today)[0][1]

    print(f"Prediction for {today.isoformat()} at ({args.lat}, {args.lon}):")
    print(f"  Predicted max temperature: {pred_temp:.1f} °C")
    print(f"  Predicted chance of rain:  {pred_rain_prob:.0%}")


if __name__ == "__main__":
    main()
