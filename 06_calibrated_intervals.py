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
CALIBRATION_SEASON = "2023-24"         # only used to learn typical errors
TEST_SEASONS = ["2024-25", "2025-26"]  # scored
LOW_Q, HIGH_Q = 0.10, 0.90             # 80% interval

def ensemble_log_forecast(i, h):
    """Average of ridge and gradient boosting, h weeks after week i, on the log scale."""
    target = y.shift(-h) - y
    train_idx = list(range(3, i - h + 1))
    X_train, y_train = X.loc[train_idx, FEATURES], target.loc[train_idx]
    x_now = X.loc[[i], FEATURES]
    ridge = Ridge(alpha=1.0).fit(X_train, y_train).predict(x_now)[0]
    boost = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3,
                                          random_state=42).fit(X_train, y_train).predict(x_now)[0]
    counts = [max(np.expm1(y[i] + change), 0) for change in (ridge, boost)]
    return np.log1p(np.mean(counts))

# 1. Point forecasts for the calibration and test seasons
rows = []
for i in flu.index[flu["season"].isin([CALIBRATION_SEASON] + TEST_SEASONS)]:
    if not 9 <= flu.loc[i, "season_week"] <= 40:
        continue
    for h in HORIZONS:
        j = i + h
        if j >= len(flu) or flu.loc[j, "imputed"]:
            continue
        rows.append({"season": flu.loc[i, "season"], "origin_week": flu.loc[i, "week"],
                     "target_week": flu.loc[j, "week"], "horizon": h,
                     "actual": flu.loc[j, "flu_filled"], "persistence": flu.loc[i, "flu_filled"],
                     "forecast_log": ensemble_log_forecast(i, h)})
bt = pd.DataFrame(rows)
bt["log_error"] = np.log1p(bt["actual"]) - bt["forecast_log"]

# 2. Interval for each forecast = spread of errors from earlier forecasts already known by then
def interval(forecast_log, horizon, as_of):
    past = bt[(bt["horizon"] == horizon) & (bt["target_week"] <= as_of)]["log_error"]
    low, high = past.quantile(LOW_Q), past.quantile(HIGH_Q)
    return max(np.expm1(forecast_log + low), 0), np.expm1(forecast_log + high), len(past)

test = bt[bt["season"].isin(TEST_SEASONS)].copy()
bounds = [interval(r.forecast_log, r.horizon, r.origin_week) for r in test.itertuples()]
test["lower_80"], test["upper_80"], test["past_errors_used"] = zip(*bounds)
test["forecast"] = np.expm1(test["forecast_log"])

summary = pd.DataFrame({
    "persistence_MAE": (test["persistence"] - test["actual"]).abs().groupby(test["horizon"]).mean(),
    "ensemble_MAE": (test["forecast"] - test["actual"]).abs().groupby(test["horizon"]).mean(),
    "coverage_%": ((test["actual"] >= test["lower_80"]) & (test["actual"] <= test["upper_80"]))
                  .groupby(test["horizon"]).mean() * 100,
    "median_width": (test["upper_80"] - test["lower_80"]).groupby(test["horizon"]).median(),
}).round(1)
print("Backtest with calibrated 80% intervals (2024-25 and 2025-26):")
print(summary.to_string())

# 3. Real forecast from the latest week, with intervals from all past errors
last = len(flu) - 1
latest_week = flu.loc[last, "week"]
future = []
for h in HORIZONS:
    f_log = ensemble_log_forecast(last, h)
    low, high, _ = interval(f_log, h, latest_week)
    future.append({"forecast_made": latest_week.date(),
                   "target_week": (latest_week + pd.Timedelta(weeks=h)).date(), "horizon": h,
                   "forecast": round(np.expm1(f_log)), "lower_80": round(low), "upper_80": round(high)})
future = pd.DataFrame(future)
print(f"\nLatest reported week: {latest_week.date()} ({flu.loc[last, 'flu_filled']:.0f} admissions)")
print(future.to_string(index=False))

# 4. Update the Tableau files
out = test[["season", "origin_week", "target_week", "horizon", "actual", "forecast",
            "lower_80", "upper_80", "persistence"]].copy()
out["origin_week"], out["target_week"] = out["origin_week"].dt.date, out["target_week"].dt.date
out.round(1).to_csv("data/tableau_backtest.csv", index=False)
future.to_csv("data/tableau_forecast.csv", index=False)
print("\nUpdated tableau_backtest.csv and tableau_forecast.csv.")