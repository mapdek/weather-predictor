# Weather Predictor

Predicts **today's** max temperature and rain probability using a
Random Forest trained on recent historical weather data.

Since today's own weather isn't known until the day is over, "predicting
today" honestly means: using yesterday's and recent days' observed
weather (lag features) plus seasonal signal (day-of-year) to estimate
today's max temperature and chance of rain. This is a real, standard
technique called **nowcasting**.

## How it works

```
Open-Meteo Archive API  --->  fetch_data.py  --->  data/history.csv
                                                        |
                                                        v
                                          features.py (lag + seasonal features)
                                                        |
                                                        v
                                              train_model.py
                                                        |
                                                        v
                                    models/temp_model.joblib, rain_model.joblib
                                                        |
                                                        v
                                              predict_today.py  ---> today's prediction
```

- **Data**: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) — free, no API key, daily max/min temp, precipitation, wind, humidity, pressure, by latitude/longitude.
- **Model**: `RandomForestRegressor` for max temperature, `RandomForestClassifier` for rain yes/no. Trained on lagged values (t-1, t-2, t-3 days) plus a cyclical day-of-year feature to capture seasonality.
- **Baseline check**: the training script always compares against a "today = yesterday" persistence baseline, so you can see whether the model is actually adding value (it should beat it on real data — lag/seasonal signal is genuinely informative for temperature; rain is much harder and may only slightly beat a majority-class baseline, which is expected for daily rain/no-rain prediction from limited features).

## Project structure

```
weather-predictor/
├── src/
│   ├── fetch_data.py      # pulls historical data from Open-Meteo
│   ├── features.py        # lag + seasonal feature engineering (shared)
│   ├── train_model.py     # trains + evaluates + saves models
│   └── predict_today.py   # fetches recent days, predicts today
├── data/                  # historical CSV lands here
├── models/                # trained .joblib models land here
├── logs/predictions.log   # daily prediction history (auto-committed)
├── .github/workflows/
│   ├── daily_predict.yml    # runs predict_today.py every morning
│   └── weekly_retrain.yml   # refetches data + retrains weekly
└── requirements.txt
```

## 1. Local setup

```bash
git clone <your-repo-url>
cd weather-predictor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Get training data

Pick a location (default is Paris; change `--lat`/`--lon` for anywhere else)
and a date range — 2–3 years of history works well:

```bash
cd src
python fetch_data.py --lat 48.8566 --lon 2.3522 \
    --start 2022-01-01 --end 2025-08-26 --out ../data/history.csv
```

## 3. Train the models

```bash
python train_model.py --data ../data/history.csv --out-dir ../models
```

This prints:
- Held-out MAE for temperature (compared against the persistence baseline)
- Held-out accuracy/F1 for rain (compared against the majority-class baseline)
- Top 5 most important features

The split is **time-based** (train on the past, test on the most recent
20% of days) — never shuffle weather time series randomly, or you leak
future information into training.

## 4. Predict today's weather

```bash
python predict_today.py --lat 48.8566 --lon 2.3522 --models-dir ../models
```

Example output:
```
Prediction for 2026-08-27 at (48.8566, 2.3522):
  Predicted max temperature: 24.3 °C
  Predicted chance of rain:  18%
```

## 5. Put it on GitHub

```bash
cd weather-predictor
git init
git add .
git commit -m "Initial weather predictor"
git branch -M main
git remote add origin https://github.com/<your-username>/weather-predictor.git
git push -u origin main
```

Then commit your trained `models/*.joblib` files too (or let the weekly
retrain workflow generate them on first run):

```bash
git add models/*.joblib data/history.csv
git commit -m "Add trained models and training data"
git push
```

## 6. Deploy: run it automatically every day

The repo includes two GitHub Actions workflows (already in
`.github/workflows/`) — nothing extra to configure, they activate as
soon as the repo is pushed with **Actions enabled**:

- **`daily_predict.yml`** — runs every day at 06:00 UTC, predicts today's
  weather, and commits the result to `logs/predictions.log`. You'll see
  a running history of predictions build up in that file over time.
- **`weekly_retrain.yml`** — runs every Sunday, refetches the latest
  data and retrains the models so they stay current.

To verify it works before waiting for the schedule: go to your repo's
**Actions** tab → select a workflow → **Run workflow** (manual trigger,
enabled via `workflow_dispatch` in both files).

### Notes on GitHub Actions permissions
Both workflows commit back to the repo, so they need write permission.
If your org has restricted default token permissions, go to
**Settings → Actions → General → Workflow permissions** and select
"Read and write permissions."

## Tracing an individual tree

The forest is 300 trees, which is too many to inspect by eye — but you can
pull out any single one and see exactly what it learned, both as text and
as a diagram:

```bash
cd src
python export_tree.py --model ../models/temp_model.joblib --tree-index 0 --max-depth 3
```

This writes two files to `logs/`:
- `tree_trace_0.txt` — the actual yes/no questions that tree learned (e.g.
  "is the day-of-year signal below X? then check yesterday's wind speed...")
- `tree_trace_0.png` — the same thing as a visual diagram

Use `--tree-index` (0 to 299) to look at a different tree, and `--max-depth`
to show more or fewer levels (trees get very wide past depth 4-5).

## A second kind of trace: model history over time

Because `weekly_retrain.yml` commits `models/*.joblib` back to the repo every
week, your git history is itself a timeline of every version of the forest
that has ever existed. To see how the model has changed over time:

```bash
git log --oneline -- models/
```

Each commit there is a fully working snapshot of the model as it existed
that week — you can check out any past commit and load that exact forest
with `joblib.load()` if you ever want to compare an old model's behavior
to the current one.

## Extending this

- **Better models**: try gradient boosting (XGBoost/LightGBM) or a small
  neural net once you have more data; compare MAE against the Random
  Forest baseline before switching.
- **More features**: cloud cover, dew point, past 7-day rolling averages,
  or pulling in a real short-range NWP forecast (e.g. Open-Meteo's
  forecast endpoint) as an additional input — that would blend
  statistical nowcasting with an actual physics-based forecast.
- **Multiple locations**: parameterize the workflows with a matrix of
  lat/lon pairs to track several cities at once.
- **Serving**: wrap `predict_today.py`'s logic in a small Flask/FastAPI
  endpoint if you want an API instead of a scheduled log file.

## Honesty about limitations

- This predicts **today's** weather using yesterday's data — it is not
  a substitute for real numerical weather prediction (NWP) models like
  those run by national weather services, which use atmospheric physics
  simulations on supercomputers. This project uses statistical
  pattern-matching on historical lags, which works reasonably for
  temperature (weather has real day-to-day persistence and seasonality)
  but is much weaker for precipitation, which is more chaotic at short
  lead times.
- Treat the rain classifier's accuracy claims with the majority-baseline
  comparison in mind — a location where it rains 20% of the time lets a
  "never predict rain" model hit 80% accuracy while being useless.
