import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

flu = pd.read_csv("data/va_flu_clean.csv", parse_dates=["week"])
flu = flu[flu["week"] >= "2022-08-01"].reset_index(drop=True)  # skip the COVID-era seasons
y = np.log1p(flu["flu_filled"])  # log scale: models learn growth rates instead of raw counts

# Features known at the time of the forecast
X = pd.DataFrame({"lag0": y, "lag1": y.shift(1), "lag2": y.shift(2), "lag3": y.shift(3)})
X["growth1"] = X["lag0"] - X["lag1"]   # this week's growth
X["growth2"] = X["lag1"] - X["lag2"]   # last week's growth
angle = 2 * np.pi * flu["week"].dt.dayofyear / 365.25
X["season_sin"], X["season_cos"] = np.sin(angle), np.cos(angle)  # time of year
FEATURES = list(X.columns)

HORIZONS = [1, 2, 3, 4]
TEST_SEASONS = ["2024-25", "2025-26"]
MODELS = {
    "ridge": lambda: Ridge(alpha=1.0),
    "gradient_boosting": lambda: HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_depth=3, random_state=42),
}

rows = []
for h in HORIZONS:
    target = y.shift(-h) - y  # how much (in log terms) admissions change over the next h weeks
    for i in flu.index[flu["season"].isin(TEST_SEASONS)]:
        if not 9 <= flu.loc[i, "season_week"] <= 40:
            continue
        j = i + h
        if j >= len(flu) or flu.loc[j, "imputed"]:
            continue
        # Train only on examples whose outcome was already known on the forecast date
        train_idx = list(range(3, i - h + 1))
        row = {"season": flu.loc[i, "season"], "origin_week": flu.loc[i, "week"].date(),
               "horizon": h, "actual": flu.loc[j, "flu_filled"],
               "persistence": flu.loc[i, "flu_filled"]}
        for name, make_model in MODELS.items():
            model = make_model().fit(X.loc[train_idx, FEATURES], target.loc[train_idx])
            predicted_log = y[i] + model.predict(X.loc[[i], FEATURES])[0]
            row[name] = max(np.expm1(predicted_log), 0)
        rows.append(row)

bt = pd.DataFrame(rows)
bt.to_csv("data/backtest_models.csv", index=False)

model_names = ["persistence"] + list(MODELS)
mae = pd.DataFrame({m: (bt[m] - bt["actual"]).abs().groupby(bt["horizon"]).mean()
                    for m in model_names}).T.round(1)
print("Mean absolute error (admissions/week) by weeks ahead:")
print(mae.to_string())

skill = (1 - mae / mae.loc["persistence"]) * 100
print("\nImprovement over persistence (%):")
print(skill.drop("persistence").round(1).to_string())