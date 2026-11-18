import pandas as pd

history = pd.read_csv("data/tableau_history.csv", parse_dates=["week"])
backtest = pd.read_csv("data/tableau_backtest.csv")
forecast = pd.read_csv("data/tableau_forecast.csv", parse_dates=["target_week"])

# 1. Current season: actual weeks plus the forecast, in one table
season = history["season"].iloc[-1]
actual = history[history["season"] == season][["week", "flu_admissions"]].copy()
actual["type"] = "Actual"
actual = actual.rename(columns={"flu_admissions": "admissions"})
actual["lower_80"] = actual["admissions"]
actual["upper_80"] = actual["admissions"]

fc = forecast.rename(columns={"target_week": "week", "forecast": "admissions"})[
    ["week", "admissions", "lower_80", "upper_80"]]
fc["type"] = "Forecast"
bridge = actual.iloc[[-1]].assign(type="Forecast")  # connects the forecast line to the last actual week
current = pd.concat([actual, bridge, fc], ignore_index=True)
current["season"] = season
current.to_csv("data/tableau_current_season.csv", index=False)

# 2. Accuracy summary: average miss by model and weeks ahead, plus interval coverage
backtest["in_interval"] = backtest["actual"].between(backtest["lower_80"], backtest["upper_80"])
acc = backtest.groupby("horizon").apply(lambda d: pd.Series({
    "Persistence": (d["persistence"] - d["actual"]).abs().mean(),
    "Ensemble model": (d["forecast"] - d["actual"]).abs().mean(),
    "coverage_pct": d["in_interval"].mean() * 100,
})).reset_index()
acc_long = acc.melt(id_vars=["horizon", "coverage_pct"], value_vars=["Persistence", "Ensemble model"],
                    var_name="model", value_name="mean_abs_error").round(1)
acc_long.to_csv("data/tableau_accuracy.csv", index=False)

print(f"Current season ({season}):")
print(current.to_string(index=False))
print("\nAccuracy:")
print(acc_long.to_string(index=False))