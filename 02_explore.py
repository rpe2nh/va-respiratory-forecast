import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/va_respiratory_raw.csv", parse_dates=["weekendingdate"])
flu = (df[["weekendingdate", "totalconfflunewadm"]]
       .rename(columns={"weekendingdate": "week", "totalconfflunewadm": "flu"})
       .sort_values("week").reset_index(drop=True))

# 1. Data quality: missing values and gaps in the weekly sequence
print("Weeks with missing flu counts:", flu["flu"].isna().sum())
gaps = flu["week"].diff().dt.days
print("Gaps longer than 7 days:", (gaps > 7).sum())
print("Weeks reporting exactly 0 flu admissions:", (flu["flu"] == 0).sum())

# 2. Label each flu season (August through July)
start_year = flu["week"].dt.year.where(flu["week"].dt.month >= 8, flu["week"].dt.year - 1)
flu["season"] = start_year.astype(str) + "-" + (start_year + 1).astype(str).str[-2:]
flu["season_week"] = flu.groupby("season").cumcount() + 1

# 3. Peak of each season
peaks = flu.loc[flu.groupby("season")["flu"].idxmax(), ["season", "week", "flu"]]
peaks["week"] = peaks["week"].dt.date
print("\nPeak week of each season:")
print(peaks.rename(columns={"week": "peak_week", "flu": "peak_admissions"}).to_string(index=False))

os.makedirs("figures", exist_ok=True)
flu.to_csv("data/va_flu_weekly.csv", index=False)

# 4. Chart: full timeline
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(flu["week"], flu["flu"], color="#AA0000")
ax.set_title("Weekly Flu Hospital Admissions in Virginia")
ax.set_ylabel("Admissions per week")
fig.tight_layout()
fig.savefig("figures/flu_timeline.png", dpi=150)

# 5. Chart: seasons overlaid so they can be compared
fig, ax = plt.subplots(figsize=(11, 5))
for season, group in flu.groupby("season"):
    current = season == flu["season"].iloc[-1]
    ax.plot(group["season_week"], group["flu"], label=season,
            linewidth=3 if current else 1.5, color="black" if current else None)
ax.set_title("Virginia Flu Admissions by Season (week 1 = early August)")
ax.set_xlabel("Week of season")
ax.set_ylabel("Admissions per week")
ax.legend()
fig.tight_layout()
fig.savefig("figures/flu_by_season.png", dpi=150)

print("\nCharts saved to figures/.")