import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # harmless Apple math-library warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

flu_all = pd.read_csv("data/va_flu_clean.csv", parse_dates=["week"])
flu = flu_all[flu_all["week"] >= "2022-08-01"].reset_index(drop=True)
y = np.log1p(flu["flu_filled"])

X = pd.DataFrame({"lag0": y, "lag1": y.shift(1), "lag2": y.shift(2), "lag3": y.shift(3)})
X["growth1"] = X["lag0"] - X["lag1"]
X["growth2"] = X["lag1"] - X["lag2"]
angle = 2 * np.pi * flu["week"].dt.dayofyear / 365.25
X["season_sin"], X["season_cos"] = np.sin(angle), np.cos(angle)
FEATURES = list(X.columns)

HORIZONS = [1, 2, 3, 4]
TEST_SEASONS = ["2024-25", "2025-26"]

def boosting(**options):
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3,
                                         random_state=42, **options)

def forecast(i, h):
    """Forecast h weeks after week i, training only on outcomes known by week i."""
    target = y.shift(-h) - y
    train_idx = list(range(3, i - h + 1))
    X_train, y_train = X.loc[train_idx, FEATURES], target.loc[train_idx]
    x_now = X.loc[[i], FEATURES]
    to_count = lambda change: max(np.expm1(y[i] + change), 0)

    ridge = to_count(Ridge(alpha=1.0).fit(X_train, y_train).predict(x_now)[0])
    boost = to_count(boosting().fit(X_train, y_train).predict(x_now)[0])
    low = to_count(boosting(loss="quantile", quantile=0.10).fit(X_train, y_train).predict(x_now)[0])
    high = to_count(boosting(loss="quantile", quantile=0.90).fit(X_train, y_train).predict(x_now)[0])

    point = (ridge + boost) / 2  # ensemble: average of the two models
    return point, min(low, point), max(high, point)

# 1. Backtest the ensemble and its 80% intervals on past seasons
rows = []
for i in flu.index[flu["season"].isin(TEST_SEASONS)]:
    if not 9 <= flu.loc[i, "season_week"] <= 40:
        continue
    for h in HORIZONS:
        j = i + h
        if j >= len(flu) or flu.loc[j, "imputed"]:
            continue
        point, low, high = forecast(i, h)
        rows.append({"season": flu.loc[i, "season"], "origin_week": flu.loc[i, "week"].date(),
                     "target_week": flu.loc[j, "week"].date(), "horizon": h,
                     "actual": flu.loc[j, "flu_filled"], "forecast": round(point, 1),
                     "lower_80": round(low, 1), "upper_80": round(high, 1),
                     "persistence": flu.loc[i, "flu_filled"]})
bt = pd.DataFrame(rows)

summary = pd.DataFrame({
    "persistence_MAE": (bt["persistence"] - bt["actual"]).abs().groupby(bt["horizon"]).mean(),
    "ensemble_MAE": (bt["forecast"] - bt["actual"]).abs().groupby(bt["horizon"]).mean(),
    "interval_coverage_%": ((bt["actual"] >= bt["lower_80"]) & (bt["actual"] <= bt["upper_80"]))
                           .groupby(bt["horizon"]).mean() * 100,
}).round(1)
summary["improvement_%"] = ((1 - summary["ensemble_MAE"] / summary["persistence_MAE"]) * 100).round(1)
print("Backtest results (2024-25 and 2025-26 seasons):")
print(summary.to_string())

# 2. Real forecast from the most recent week
last = len(flu) - 1
latest_week = flu.loc[last, "week"]
future = []
for h in HORIZONS:
    point, low, high = forecast(last, h)
    future.append({"forecast_made": latest_week.date(),
                   "target_week": (latest_week + pd.Timedelta(weeks=h)).date(), "horizon": h,
                   "forecast": round(point), "lower_80": round(low), "upper_80": round(high)})
future = pd.DataFrame(future)
print(f"\nLatest reported week: {latest_week.date()} ({flu.loc[last, 'flu_filled']:.0f} admissions)")
print("Forecast:")
print(future.to_string(index=False))

# 3. Files for the Tableau dashboard
flu_all[["week", "season", "season_week", "flu_filled", "imputed"]].rename(
    columns={"flu_filled": "flu_admissions"}).to_csv("data/tableau_history.csv", index=False)
bt.to_csv("data/tableau_backtest.csv", index=False)
future.to_csv("data/tableau_forecast.csv", index=False)
print("\nSaved tableau_history.csv, tableau_backtest.csv, and tableau_forecast.csv to data/.")