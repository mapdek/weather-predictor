"""
train_model.py
Trains two models on historical daily weather data:
  1. RandomForestRegressor  -> predicts today's max temperature
  2. RandomForestClassifier -> predicts whether it rains today (precip > 0)

Uses a time-based train/test split (never shuffle time series data --
that would leak the future into training).
"""

import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score

from features import build_features, feature_columns


def main():
    parser = argparse.ArgumentParser(description="Train weather prediction models")
    parser.add_argument("--data", type=str, default="data/history.csv")
    parser.add_argument("--test-frac", type=float, default=0.2, help="Fraction of most recent data held out for testing")
    parser.add_argument("--out-dir", type=str, default="models")
    args = parser.parse_args()

    raw = pd.read_csv(args.data, parse_dates=["date"])
    feat = build_features(raw)
    cols = feature_columns(feat)

    split_idx = int(len(feat) * (1 - args.test_frac))
    train, test = feat.iloc[:split_idx], feat.iloc[split_idx:]
    print(f"Train rows: {len(train)}  Test rows: {len(test)}")

    X_train, X_test = train[cols], test[cols]

    # --- Regression: max temperature ---
    reg = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    reg.fit(X_train, train["target_temp_max"])
    pred_temp = reg.predict(X_test)
    mae = mean_absolute_error(test["target_temp_max"], pred_temp)
    print(f"[Temperature] MAE on held-out test set: {mae:.2f} °C")

    # baseline for comparison: "today = yesterday" (persistence model)
    persistence_mae = mean_absolute_error(test["target_temp_max"], test["temperature_2m_max_lag1"])
    print(f"[Temperature] Persistence baseline MAE (today=yesterday): {persistence_mae:.2f} °C")

    # --- Classification: rain yes/no ---
    clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
    clf.fit(X_train, train["target_rain"])
    pred_rain = clf.predict(X_test)
    acc = accuracy_score(test["target_rain"], pred_rain)
    f1 = f1_score(test["target_rain"], pred_rain)
    print(f"[Rain] Accuracy: {acc:.2%}  F1: {f1:.2f}")

    rain_rate = test["target_rain"].mean()
    baseline_acc = max(rain_rate, 1 - rain_rate)
    print(f"[Rain] Majority-class baseline accuracy: {baseline_acc:.2%}")

    # --- feature importance (quick sanity check) ---
    importances = pd.Series(reg.feature_importances_, index=cols).sort_values(ascending=False)
    print("\nTop 5 features for temperature prediction:")
    print(importances.head(5).to_string())

    # --- save models ---
    joblib.dump({"model": reg, "features": cols}, f"{args.out_dir}/temp_model.joblib")
    joblib.dump({"model": clf, "features": cols}, f"{args.out_dir}/rain_model.joblib")
    print(f"\nSaved models to {args.out_dir}/")


if __name__ == "__main__":
    main()
