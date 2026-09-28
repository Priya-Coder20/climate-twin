import os
import glob

import numpy as np
import xarray as xr
import rasterio
import geopandas as gpd
from shapely.geometry import Point


PROJECT = r"C:\Users\HP\climate-twin"

MOSDAC_DIR = os.path.join(
    PROJECT, r"data\processed_daily\combined"
)

ERA5_FILE = os.path.join(
    PROJECT, r"data\processed_daily\era5\era5_daily.nc"
)

IMD_TEMP_FILE = os.path.join(
    PROJECT, r"data\processed_daily\imd\imd_temperature_daily.nc"
)

IMD_RAIN_FILE = os.path.join(
    PROJECT, r"data\processed_daily\imd\imd_rainfall_daily.nc"
)

BOUNDARY_FILE = os.path.join(
    PROJECT, r"data\processed_daily\maharashtra_boundary_wgs84.shp"
)

OUTPUT_DIR = os.path.join(
    PROJECT, r"data\processed_daily\cnn_dataset"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, "common_daily_v2.nc"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load datasets
era5 = xr.open_dataset(ERA5_FILE)
imd_temp = xr.open_dataset(IMD_TEMP_FILE)
imd_rain = xr.open_dataset(IMD_RAIN_FILE)


# Maharashtra boundary
maharashtra = gpd.read_file(BOUNDARY_FILE).to_crs("EPSG:4326")
geometry = maharashtra.geometry.union_all()


# Find common dates
mosdac_folders = sorted(
    glob.glob(os.path.join(MOSDAC_DIR, "*"))
)

mosdac_dates = []

for folder in mosdac_folders:
    if os.path.isdir(folder):
        if (
            os.path.exists(os.path.join(folder, "imr.tif"))
            and os.path.exists(os.path.join(folder, "pet_aligned.tif"))
        ):
            mosdac_dates.append(os.path.basename(folder))

era5_dates = set(str(x)[:10] for x in era5.valid_time.values)
imd_dates = set(str(x)[:10] for x in imd_temp.time.values)

common_dates = sorted(
    set(mosdac_dates) & era5_dates & imd_dates
)

if not common_dates:
    raise ValueError("No common dates found.")


# Target grid from MOSDAC
first_imr = os.path.join(
    MOSDAC_DIR, common_dates[0], "imr.tif"
)

with rasterio.open(first_imr) as src:
    height = src.height
    width = src.width
    transform = src.transform
    bounds = src.bounds

target_lon = (
    bounds.left
    + (np.arange(width) + 0.5) * transform.a
)

target_lat = (
    bounds.top
    + (np.arange(height) + 0.5) * transform.e
)

target_lat = target_lat[::-1]


# Maharashtra mask on final grid
mask = np.zeros(
    (len(target_lat), len(target_lon)),
    dtype=bool
)

for i, lat in enumerate(target_lat):
    for j, lon in enumerate(target_lon):
        point = Point(float(lon), float(lat))

        if geometry.contains(point) or geometry.touches(point):
            mask[i, j] = True

mask_da = xr.DataArray(
    mask,
    coords={
        "latitude": target_lat,
        "longitude": target_lon
    },
    dims=["latitude", "longitude"]
)


# Process daily data
data_arrays = []

for date in common_dates:

    imr_file = os.path.join(
        MOSDAC_DIR, date, "imr.tif"
    )

    pet_file = os.path.join(
        MOSDAC_DIR, date, "pet_aligned.tif"
    )

    with rasterio.open(imr_file) as src:
        imr = src.read(1).astype(np.float32)

    with rasterio.open(pet_file) as src:
        pet = src.read(1).astype(np.float32)

    imr = np.flipud(imr)
    pet = np.flipud(pet)


    # ERA5
    era5_day = era5.sel(valid_time=date)

    era5_day = era5_day.interp(
        latitude=target_lat,
        longitude=target_lon,
        method="linear"
    )


    # IMD temperature
    imd_temp_day = imd_temp.sel(time=date)

    imd_temp_day = imd_temp_day.interp(
        latitude=target_lat,
        longitude=target_lon,
        method="linear"
    )


    # IMD rainfall
    imd_rain_day = imd_rain.sel(time=date)

    imd_rain_day = imd_rain_day.interp(
        latitude=target_lat,
        longitude=target_lon,
        method="linear"
    )


    # Daily dataset
    daily = xr.Dataset(
        {
            "imr": (
                ["latitude", "longitude"],
                imr
            ),

            "pet": (
                ["latitude", "longitude"],
                pet
            ),

            "era5_temperature": (
                ["latitude", "longitude"],
                era5_day.temperature_c.values
            ),

            "era5_dewpoint": (
                ["latitude", "longitude"],
                era5_day.dewpoint_c.values
            ),

            "era5_wind_speed": (
                ["latitude", "longitude"],
                era5_day.wind_speed.values
            ),

            "era5_pressure": (
                ["latitude", "longitude"],
                era5_day.pressure_hpa.values
            ),

            "era5_rainfall": (
                ["latitude", "longitude"],
                era5_day.rainfall_mm.values
            ),

            "imd_max_temperature": (
                ["latitude", "longitude"],
                imd_temp_day.max_temperature.values
            ),

            "imd_min_temperature": (
                ["latitude", "longitude"],
                imd_temp_day.min_temperature.values
            ),

            "imd_rainfall": (
                ["latitude", "longitude"],
                imd_rain_day.rainfall.values
            )
        },

        coords={
            "latitude": target_lat,
            "longitude": target_lon
        }
    )


    # Apply Maharashtra mask AFTER regridding
    daily = daily.where(mask_da)

    daily = daily.expand_dims(
        time=[np.datetime64(date)]
    )

    data_arrays.append(daily)


# Combine
cnn_dataset = xr.concat(
    data_arrays,
    dim="time"
)


# Save
cnn_dataset.to_netcdf(OUTPUT_FILE)


era5.close()
imd_temp.close()
imd_rain.close()


print("CNN dataset created.")
print("Days:", cnn_dataset.sizes["time"])
print("Output:", OUTPUT_FILE)