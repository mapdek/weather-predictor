"""
fetch_data.py
Downloads historical daily weather data from the Open-Meteo Archive API
(free, no API key required) for a given location and date range.

API docs: https://open-meteo.com/en/docs/historical-weather-api
"""

import argparse
import sys
import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
]


def fetch_history(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily historical weather for a location between two ISO dates."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARS),
        "timezone": "auto",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    if "daily" not in payload:
        raise RuntimeError(f"Unexpected API response: {payload}")

    df = pd.DataFrame(payload["daily"])
    df.rename(columns={"time": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch historical daily weather data")
    parser.add_argument("--lat", type=float, default=48.8566, help="Latitude (default: Paris)")
    parser.add_argument("--lon", type=float, default=2.3522, help="Longitude (default: Paris)")
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--out", type=str, default="data/history.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"Fetching data for ({args.lat}, {args.lon}) from {args.start} to {args.end}...")
    df = fetch_history(args.lat, args.lon, args.start, args.end)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
