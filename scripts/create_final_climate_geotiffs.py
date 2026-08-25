import os
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_bounds

PROJECT = r"C:\Users\HP\climate-twin"

INPUT = os.path.join(
    PROJECT,
    r"data\processed_daily\cnn_dataset\common_daily.nc"
)

OUTPUT_DIR = os.path.join(
    PROJECT,
    r"data\processed_daily\final_climate_geotiffs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

ds = xr.open_dataset(INPUT)

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

band_names = [
    "IMR",
    "PET",
    "ERA5_Temperature",
    "ERA5_Dewpoint",
    "ERA5_Wind_Speed",
    "ERA5_Pressure",
    "ERA5_Rainfall",
    "IMD_Max_Temperature",
    "IMD_Min_Temperature",
    "IMD_Rainfall"
]

lat = ds.latitude.values
lon = ds.longitude.values

height = len(lat)
width = len(lon)

transform = from_bounds(
    float(lon.min()),
    float(lat.min()),
    float(lon.max()),
    float(lat.max()),
    width,
    height
)

for i, time in enumerate(ds.time.values):

    date = str(time)[:10]

    output = os.path.join(
        OUTPUT_DIR,
        f"final_climate_{date}.tif"
    )

    data = []

    for variable in variables:
        array = ds[variable].isel(time=i).values

        array = np.nan_to_num(
            array,
            nan=-999.0
        )

        data.append(
            array.astype(np.float32)
        )

    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=10,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-999.0,
        compress="deflate"
    ) as dst:

        for band, array in enumerate(data, start=1):
            dst.write(array, band)
            dst.set_band_description(
                band,
                band_names[band - 1]
            )

ds.close()

print("Final climate GeoTIFFs created.")
print("Output:", OUTPUT_DIR)