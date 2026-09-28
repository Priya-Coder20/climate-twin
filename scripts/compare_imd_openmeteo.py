import os
import numpy as np
import pandas as pd

IMD_FILE = r"data\live\imd_daily_maharashtra.csv"
OPENMETEO_FILE = r"data\live\openmeteo_station_daily.csv"
OUTPUT = r"data\live\imd_openmeteo_comparison.csv"

imd = pd.read_csv(IMD_FILE)
om = pd.read_csv(OPENMETEO_FILE)

imd["date"] = pd.to_datetime(imd["date"]).dt.strftime("%Y-%m-%d")
om["date"] = pd.to_datetime(om["date"]).dt.strftime("%Y-%m-%d")

for col in ["tmax", "tmin", "rainfall"]:
    imd[col] = pd.to_numeric(imd[col], errors="coerce")
    om[col] = pd.to_numeric(om[col], errors="coerce")

imd = (
    imd.groupby(["station", "date"], as_index=False)
    [["tmax", "tmin", "rainfall"]]
    .mean()
)

om = (
    om.groupby(["station", "date"], as_index=False)
    [["tmax", "tmin", "rainfall"]]
    .mean()
)

merged = pd.merge(
    imd,
    om,
    on=["station", "date"],
    suffixes=("_imd", "_openmeteo")
)

rows = []

for variable in ["tmax", "tmin", "rainfall"]:

    valid = merged[
        merged[f"{variable}_imd"].notna()
        & merged[f"{variable}_openmeteo"].notna()
    ].copy()

    error = (
        valid[f"{variable}_openmeteo"]
        - valid[f"{variable}_imd"]
    )

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    bias = np.mean(error)

    rows.append({
        "variable": variable,
        "samples": len(valid),
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias
    })

    merged.loc[
        valid.index,
        f"{variable}_error"
    ] = error

summary = pd.DataFrame(rows)

os.makedirs(
    os.path.dirname(OUTPUT),
    exist_ok=True
)

merged.to_csv(OUTPUT, index=False)

print()
print("IMD vs OPEN-METEO COMPARISON")
print()

print(summary.to_string(index=False))

print()
print("Stations compared:", merged["station"].nunique())
print("Station-days compared:", len(merged))
print()
print("Detailed comparison saved:")
print(OUTPUT)