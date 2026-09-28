import os
import time
import requests
import pandas as pd

INPUT = r"data\live\imd_daily_maharashtra.csv"
OUTPUT = r"data\live\openmeteo_station_daily.csv"

URL = "https://api.open-meteo.com/v1/forecast"

df = pd.read_csv(INPUT)

stations = (
    df.groupby("station")[["latitude", "longitude"]]
    .mean()
    .reset_index()
)

latitudes = stations["latitude"].tolist()
longitudes = stations["longitude"].tolist()

params = {
    "latitude": ",".join(map(str, latitudes)),
    "longitude": ",".join(map(str, longitudes)),
    "start_date": "2026-09-17",
    "end_date": "2026-09-23",
    "daily": ",".join([
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "dew_point_2m_mean",
        "pressure_msl_mean",
        "wind_speed_10m_mean",
        "precipitation_sum"
    ]),
    "timezone": "Asia/Kolkata"
}

print("Requesting Open-Meteo data...")
print("Stations:", len(stations))

response = requests.get(
    URL,
    params=params,
    timeout=60
)

print("STATUS:", response.status_code)

response.raise_for_status()

data = response.json()

if isinstance(data, dict):
    data = [data]

records = []

for station, result in zip(stations["station"], data):

    daily = result.get("daily", {})

    dates = daily.get("time", [])

    for i, date in enumerate(dates):

        records.append({
            "station": station,
            "latitude": result["latitude"],
            "longitude": result["longitude"],
            "date": date,
            "temperature": daily["temperature_2m_mean"][i],
            "tmax": daily["temperature_2m_max"][i],
            "tmin": daily["temperature_2m_min"][i],
            "dewpoint": daily["dew_point_2m_mean"][i],
            "pressure": daily["pressure_msl_mean"][i],
            "wind_speed": daily["wind_speed_10m_mean"][i],
            "rainfall": daily["precipitation_sum"][i]
        })

result_df = pd.DataFrame(records)

os.makedirs(
    os.path.dirname(OUTPUT),
    exist_ok=True
)

result_df.to_csv(
    OUTPUT,
    index=False
)

print()
print("==============================")
print("DOWNLOAD COMPLETED")
print("==============================")
print("Stations:", result_df["station"].nunique())
print("Dates:", result_df["date"].nunique())
print("Rows:", len(result_df))
print("Saved:", OUTPUT)
print()
print(result_df.head(10).to_string(index=False))