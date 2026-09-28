import pandas as pd
import os

INPUT = r"data\live\imd_maharashtra_2026-09-20_full.csv"
OUTPUT = r"data\live\imd_daily_maharashtra.csv"

df = pd.read_csv(INPUT)

df["value"] = pd.to_numeric(df["value"], errors="coerce")
df["report_time"] = pd.to_datetime(df["report_time"], errors="coerce")

df = df.dropna(subset=["value", "report_time"])

df["date"] = df["report_time"].dt.date

variables = {
    "air_temperature": "temperature",
    "dewpoint_temperature": "dewpoint",
    "pressure_reduced_to_mean_sea_level": "pressure",
    "total_precipitation_or_total_water_equivalent": "rainfall",
    "minimum_temperature_at_height_and_over_period_specified": "tmin",
    "maximum_temperature_at_height_and_over_period_specified": "tmax",
    "wind_speed": "wind_speed"
}

df = df[df["variable"].isin(variables)]
df["variable"] = df["variable"].map(variables)

records = []

for (date, station, lat, lon), group in df.groupby(
    ["date", "station", "latitude", "longitude"]
):

    row = {
        "date": date,
        "station": station,
        "latitude": lat,
        "longitude": lon
    }

    for variable, values in group.groupby("variable"):

        if variable == "temperature":
            row[variable] = values["value"].mean()

        elif variable == "dewpoint":
            row[variable] = values["value"].mean()

        elif variable == "pressure":
            row[variable] = values["value"].mean()

        elif variable == "rainfall":
            row[variable] = values["value"].sum()

        elif variable == "tmin":
            row[variable] = values["value"].min()

        elif variable == "tmax":
            row[variable] = values["value"].max()

        elif variable == "wind_speed":
            row[variable] = values["value"].mean()

    records.append(row)

daily = pd.DataFrame(records)

os.makedirs(
    os.path.dirname(OUTPUT),
    exist_ok=True
)

daily.to_csv(
    OUTPUT,
    index=False
)

print("Input observations:", len(df))
print("Daily station records:", len(daily))
print("Stations:", daily["station"].nunique())
print("Dates:", daily["date"].nunique())
print("Saved:", OUTPUT)