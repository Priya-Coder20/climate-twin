import os
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_bounds

INPUT = r"data\processed_daily\cnn_dataset\common_daily.nc"

OUTPUT = r"data\processed_daily\cnn_dataset\final_climate_2021-06-16.tif"

variables = [
    "imr",
    "pet",
    "era5_temperature",
    "era5_dewpoint",
    "era5_wind_speed",
    "era5_pressure",
    "era5_rainfall",
    "imd_max_temperature",
    "imd_min_temperature",
    "imd_rainfall"
]

ds = xr.open_dataset(INPUT)

day = ds.sel(
    time="2021-06-16"
)

lat = day.latitude.values
lon = day.longitude.values

data = []

for variable in variables:
    values = day[variable].values

    values = np.nan_to_num(
        values,
        nan=-999.0
    )

    data.append(values)

data = np.asarray(data, dtype=np.float32)

transform = from_bounds(
    float(lon.min()),
    float(lat.min()),
    float(lon.max()),
    float(lat.max()),
    len(lon),
    len(lat)
)

os.makedirs(
    os.path.dirname(OUTPUT),
    exist_ok=True
)

with rasterio.open(
    OUTPUT,
    "w",
    driver="GTiff",
    height=len(lat),
    width=len(lon),
    count=len(variables),
    dtype="float32",
    crs="EPSG:4326",
    transform=transform,
    nodata=-999.0
) as dst:

    for i, variable in enumerate(variables, start=1):
        dst.write(data[i - 1], i)
        dst.set_band_description(i, variable)

ds.close()

print("GeoTIFF created:")
print(OUTPUT)