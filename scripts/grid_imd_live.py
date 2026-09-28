import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

INPUT = r"data\live\imd_daily_maharashtra.csv"
GRID_FILE = r"data\processed_daily\cnn_dataset\common_daily_v2.nc"
OUTPUT = r"data\live\imd_gridded_daily.nc"

df = pd.read_csv(INPUT)

grid_ds = xr.open_dataset(GRID_FILE)

lat = grid_ds["latitude"].values
lon = grid_ds["longitude"].values

target_mask = np.isfinite(
    grid_ds["imd_max_temperature"].isel(time=0).values
)

df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

df["tmax"] = pd.to_numeric(df["tmax"], errors="coerce")
df["tmin"] = pd.to_numeric(df["tmin"], errors="coerce")
df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")

records = []

for date, day in df.groupby("date"):

    station_coords = (
        day.groupby("station")[["latitude", "longitude"]]
        .mean()
    )

    for variable in ["tmax", "tmin", "rainfall"]:

        values = (
            day.groupby("station")[variable]
            .mean()
            .dropna()
        )

        values = values[values.index.isin(station_coords.index)]

        if len(values) < 3:
            continue

        coords = station_coords.loc[values.index][
            ["longitude", "latitude"]
        ].values

        points = np.column_stack(
            [coords[:, 0], coords[:, 1]]
        )

        mesh_lon, mesh_lat = np.meshgrid(lon, lat)

        grid = griddata(
            points,
            values.values,
            (mesh_lon, mesh_lat),
            method="linear"
        )

        nearest = griddata(
            points,
            values.values,
            (mesh_lon, mesh_lat),
            method="nearest"
        )

        grid = np.where(
            np.isnan(grid),
            nearest,
            grid
        )

        grid = np.where(
            target_mask,
            grid,
            np.nan
        )

        records.append(
            xr.DataArray(
                grid,
                dims=["latitude", "longitude"],
                coords={
                    "latitude": lat,
                    "longitude": lon
                },
                name=variable,
                attrs={"units": "station-derived"}
            ).expand_dims(time=[np.datetime64(date)])
        )

result = xr.merge(records)

result = result.sortby("time")

os.makedirs(
    os.path.dirname(OUTPUT),
    exist_ok=True
)

result.to_netcdf(OUTPUT)

print("Saved:", OUTPUT)
print("Shape:", result.sizes)
print("Variables:", list(result.data_vars))