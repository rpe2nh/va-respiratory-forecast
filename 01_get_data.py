import os
import pandas as pd

# CDC NHSN Weekly Hospital Respiratory Data, Virginia only, oldest week first
URL = ("https://data.cdc.gov/resource/ua7e-t2fy.csv"
       "?jurisdiction=VA&$limit=50000&$order=weekendingdate")

df = pd.read_csv(URL)
os.makedirs("data", exist_ok=True)
df.to_csv("data/va_respiratory_raw.csv", index=False)

print("Rows, columns:", df.shape)
print("\nFirst 15 column names:", df.columns.tolist()[:15])

admission_cols = [c for c in df.columns if c.startswith("totalconf") and c.endswith("newadm")]
print("\nAdmission columns:", admission_cols)

date_col = "weekendingdate"
print(f"\nWeeks covered: {df[date_col].min()[:10]} to {df[date_col].max()[:10]}")
print("\nMost recent 8 weeks:")
print(df[[date_col] + admission_cols].tail(8).to_string(index=False))


