import os
import requests
import numpy as np
import pandas as pd
import xarray as xr

GRID_FILE = r"data\processed_daily\cnn_dataset\common_daily_v2.nc"
OUTPUT = r"data\live\openmeteo_maharashtra_daily.nc"

START_DATE = "2026-09-18"
END_DATE = "2026-09-24"

ds = xr.open_dataset(GRID_FILE)

lat = ds["latitude"].values
lon = ds["longitude"].values

mask = np.isfinite(
    ds["imd_max_temperature"].isel(time=0).values
)

lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

points = pd.DataFrame({
    "lat": lat_grid[mask],
    "lon": lon_grid[mask]
})

records = []

for start in range(0, len(points), 100):

    batch = points.iloc[start:start + 100]

    params = {
        "latitude": ",".join(batch["lat"].astype(str)),
        "longitude": ",".join(batch["lon"].astype(str)),
        "start_date": START_DATE,
        "end_date": END_DATE,
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

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    for i, point in enumerate(data):

        daily = point["daily"]

        for j, date in enumerate(daily["time"]):

            records.append({
                "date": date,
                "latitude": point["latitude"],
                "longitude": point["longitude"],
                "tmax": daily["temperature_2m_max"][j],
                "tmin": daily["temperature_2m_min"][j],
                "temperature": daily["temperature_2m_mean"][j],
                "dewpoint": daily["dew_point_2m_mean"][j],
                "pressure": daily["pressure_msl_mean"][j],
                "wind_speed": daily["wind_speed_10m_mean"][j],
                "rainfall": daily["precipitation_sum"][j]
            })

result = pd.DataFrame(records)

result["date"] = pd.to_datetime(result["date"])

output = result.set_index(
    ["date", "latitude", "longitude"]
).to_xarray()

os.makedirs(
    os.path.dirname(OUTPUT),
    exist_ok=True
)

output.to_netcdf(OUTPUT)

print("Saved:", OUTPUT)
print("Dates:", result["date"].nunique())
print("Grid points:", result[["latitude", "longitude"]].drop_duplicates().shape[0])
print("Variables:", list(output.data_vars))
print("Rainfall min:", result["rainfall"].min())
print("Rainfall max:", result["rainfall"].max())