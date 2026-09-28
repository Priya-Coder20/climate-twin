import pandas as pd
import os

INPUT = r"data\live\imd_2026-09-20.csv"
OUTPUT = r"data\live\imd_daily_2026-09-20.csv"

df = pd.read_csv(INPUT)

# Variables we need
wanted = [
    "air_temperature",
    "dewpoint_temperature",
    "pressure_reduced_to_mean_sea_level",
    "total_precipitation_or_total_water_equivalent",
    "minimum_temperature_at_height_and_over_period_specified",
    "maximum_temperature_at_height_and_over_period_specified",
    "wind_speed",
    "wind_direction",
]

df = df[df["variable"].isin(wanted)].copy()

# Convert values to numeric
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# Remove observations without numerical values
df = df.dropna(subset=["value"])

# Extract date
df["date"] = pd.to_datetime(
    df["report_time"],
    errors="coerce"
).dt.date

# Convert observations from long format to wide format
daily = df.pivot_table(
    index=["date", "station", "latitude", "longitude"],
    columns="variable",
    values="value",
    aggfunc="mean"
).reset_index()

daily.columns.name = None

# Rename long IMD names
daily = daily.rename(columns={
    "air_temperature": "temperature",
    "dewpoint_temperature": "dewpoint",
    "pressure_reduced_to_mean_sea_level": "pressure",
    "total_precipitation_or_total_water_equivalent": "rainfall",
    "minimum_temperature_at_height_and_over_period_specified": "tmin",
    "maximum_temperature_at_height_and_over_period_specified": "tmax",
    "wind_speed": "wind_speed",
    "wind_direction": "wind_direction",
})

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

daily.to_csv(OUTPUT, index=False)

print("IMD daily preprocessing completed.")
print("Input rows:", len(df))
print("Stations:", daily["station"].nunique())
print("Output rows:", len(daily))
print()
print(daily.to_string(index=False))
print()
print("Saved:", OUTPUT)