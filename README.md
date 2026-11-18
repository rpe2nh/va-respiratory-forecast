# Forecasting Flu Hospital Admissions in Virginia

Weekly forecasts of flu hospital admissions in Virginia, 1–4 weeks ahead, built from CDC hospital data and backtested on past seasons.

**[Live Tableau dashboard →](https://public.tableau.com/app/profile/mark.espiritu/viz/VirginiaFluForecast/VirginiaFluForecast)**

Updates automatically every week: GitHub Actions reruns the forecast, a Google Sheet pulls the new results, and Tableau Public refreshes daily.

## Key findings

- **The model beats a simple baseline by 20–37%.** In backtests on the 2024-25 and 2025-26 seasons, the average miss 4 weeks ahead was 151 admissions per week, vs. 240 for "next week = this week."
- **Naive trend extrapolation is dangerous.** Assuming this week's growth rate continues was more than 3x worse than the simple baseline 4 weeks out, because it overshoots right through the peak.
- **Peak timing varies by two months.** Recent seasons peaked anywhere from early December to February, so forecasts have to follow week-to-week growth rather than assume a calendar date.
- **Honest uncertainty matters.** The model's first 80% intervals only contained the actual value 52–63% of the time. Calibrating them on past forecast errors raised that to 66–75%.

## Approach

| Step | File | What it does |
|---|---|---|
| 1 | `01_get_data.py` | Downloads Virginia's weekly data from CDC's NHSN API |
| 2 | `02_explore.py` | Checks data quality and compares flu seasons |
| 3 | `03_baselines.py` | Fills missing weeks and backtests two simple baselines |
| 4 | `04_models.py` | Backtests ridge regression and gradient boosting, retrained every week without future data |
| 5 | `05_forecast.py` | Tests an ensemble with quantile-regression intervals |
| 6 | `06_calibrated_intervals.py` | Calibrates intervals on past forecast errors and makes the current forecast |
| 7 | `07_tableau_prep.py` | Builds the files the Tableau dashboard reads |

**Modeling choices:** models predict the change in log admissions using the last four weeks, recent growth, and the time of year. Each horizon (1–4 weeks) has its own model. The 2020-21 and 2021-22 seasons are excluded because COVID-era measures nearly eliminated flu.

## Limitations

- **Only four normal flu seasons** are available to learn from, so unusual seasons are hard to predict.
- **Intervals are still slightly overconfident** in severe seasons (66–75% coverage vs. the 80% target).
- **CDC data can be revised** for a few weeks after it's first published.
- **The forecast doesn't use outside signals** like testing rates, vaccination, or which flu strains are circulating.

## Run it yourself

```bash
pip install -r requirements.txt
python3 01_get_data.py
python3 02_explore.py
python3 03_baselines.py
python3 06_calibrated_intervals.py
python3 07_tableau_prep.py
```

## Data source

CDC National Healthcare Safety Network (NHSN), *Weekly Hospital Respiratory Data (HRD) Metrics by Jurisdiction* (dataset