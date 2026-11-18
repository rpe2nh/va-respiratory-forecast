import numpy as np
import pandas as pd

flu = pd.read_csv("data/va_flu_weekly.csv", parse_dates=["week"])

# 1. List and fill missing weeks (straight line between the neighboring weeks)
print("Missing weeks:", flu.loc[flu["flu"].isna(), "week"].dt.date.tolist())
flu["imputed"] = flu["flu"].isna()
flu["flu_filled"] = flu["flu"].interpolate(limit_direction="both")
flu.to_csv("data/va_flu_clean.csv", index=False)

# 2. Backtest: pretend it's each week of past seasons and forecast 1-4 weeks ahead
HORIZONS = [1, 2, 3, 4]
TEST_SEASONS = ["2024-25", "2025-26"]

rows = []
for i in flu.index[flu["season"].isin(TEST_SEASONS)]:
    if not 9 <= flu.loc[i, "season_week"] <= 40:  # roughly October through April
        continue
    last = flu.loc[i, "flu_filled"]
    prev = flu.loc[i - 1, "flu_filled"]
    growth = np.clip(last / prev if prev > 0 else 1.0, 0.5, 2.0)  # cap wild week-to-week swings
    for h in HORIZONS:
        j = i + h
        if j >= len(flu) or flu.loc[j, "imputed"]:
            continue  # only score forecasts against real reported weeks
        rows.append({
            "season": flu.loc[i, "season"],
            "origin_week": flu.loc[i, "week"].date(),
            "horizon": h,
            "actual": flu.loc[j, "flu_filled"],
            "persistence": last,               # "next weeks = this week"
            "trend": last * growth ** h,       # "keep growing at this week's rate"
        })

bt = pd.DataFrame(rows)
bt.to_csv("data/backtest_baselines.csv", index=False)

# 3. Score: mean absolute error (average miss, in admissions per week)
mae = (bt.melt(id_vars=["season", "horizon", "actual"], value_vars=["persistence", "trend"],
               var_name="model", value_name="forecast")
         .assign(abs_error=lambda d: (d["forecast"] - d["actual"]).abs())
         .groupby(["model", "horizon"])["abs_error"].mean()
         .unstack("horizon").round(1))
print(f"\nBacktest forecasts: {len(bt)} across {', '.join(TEST_SEASONS)}")
print("\nMean absolute error (admissions/week) by weeks ahead:")
print(mae.to_string())