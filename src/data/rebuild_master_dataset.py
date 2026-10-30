import os
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point

# PATHS

ERA5_FILE = r"data\processed_daily\era5\era5_daily.nc"
IMD_RAIN_FILE = r"data\processed_daily\imd_maharashtra\imd_rainfall_maharashtra_daily.nc"
IMD_TEMP_FILE = r"data\processed_daily\imd_maharashtra\imd_temperature_maharashtra_daily.nc"

OLD_MASTER_FILE = r"data\processed_daily\cnn_dataset\common_daily_v2.nc"
BOUNDARY_FILE = r"data\boundary\maharashtra_wgs84.shp"

OUTPUT_FILE = r"data\processed\master_climate_dataset_v2.nc"

# LOAD DATA

print("REBUILDING MAHARASHTRA CLIMATE MASTER DATASET")

era5 = xr.open_dataset(ERA5_FILE)
imd_rain = xr.open_dataset(IMD_RAIN_FILE)
imd_temp = xr.open_dataset(IMD_TEMP_FILE)
old = xr.open_dataset(OLD_MASTER_FILE)
boundary = gpd.read_file(BOUNDARY_FILE)

if boundary.crs is None:
    raise ValueError("Boundary CRS is missing.")

boundary = boundary.to_crs("EPSG:4326")

# STANDARDIZE TIME

era5 = era5.rename({"valid_time": "time"})

# Convert to daily datetime64
era5["time"] = era5.time.astype("datetime64[ns]")
imd_rain["time"] = imd_rain.time.astype("datetime64[ns]")
imd_temp["time"] = imd_temp.time.astype("datetime64[ns]")
old["time"] = old.time.astype("datetime64[ns]")

# Normalize to date
era5["time"] = era5.time.dt.floor("D")
imd_rain["time"] = imd_rain.time.dt.floor("D")
imd_temp["time"] = imd_temp.time.dt.floor("D")
old["time"] = old.time.dt.floor("D")

# FIND COMMON DATES

print("\nFinding common dates...")

common_era5_imd = np.intersect1d(
    era5.time.values,
    imd_rain.time.values
)

common_dates = np.intersect1d(
    common_era5_imd,
    imd_temp.time.values
)

# Restrict to historical period represented by old MOSDAC data
common_dates = np.intersect1d(
    common_dates,
    old.time.values
)

print("Common dates:", len(common_dates))

if len(common_dates) == 0:
    raise ValueError("No common dates found.")

print("First:", common_dates[0])
print("Last :", common_dates[-1])

# SELECT COMMON PERIOD
era5 = era5.sel(time=common_dates)
imd_rain = imd_rain.sel(time=common_dates)
imd_temp = imd_temp.sel(time=common_dates)
old = old.sel(time=common_dates)

# TARGET GRID
target_lat = old.latitude
target_lon = old.longitude

print("Latitude:", len(target_lat))
print("Longitude:", len(target_lon))

# SPATIAL REGRIDDING
era5_target = era5[
    [
        "temperature_c",
        "dewpoint_c",
        "wind_speed",
        "pressure_hpa",
        "rainfall_mm",
    ]
].interp(
    latitude=target_lat,
    longitude=target_lon,
    method="linear"
)


imd_rain_target = imd_rain[
    ["rainfall"]
].interp(
    latitude=target_lat,
    longitude=target_lon,
    method="linear"
)


imd_temp_target = imd_temp[
    [
        "max_temperature",
        "min_temperature",
    ]
].interp(
    latitude=target_lat,
    longitude=target_lon,
    method="linear"
)

# EXTRACT MOSDAC VARIABLES FROM EXISTING MASTER
required_mosdac = ["imr", "pet"]

for var in required_mosdac:
    if var not in old:
        raise ValueError(f"MOSDAC variable missing from old master: {var}")

mosdac = old[required_mosdac]

# COMBINE VARIABLES
master = xr.Dataset(
    {
        "imr": mosdac["imr"],
        "pet": mosdac["pet"],

        "era5_temperature": era5_target["temperature_c"],
        "era5_dewpoint": era5_target["dewpoint_c"],
        "era5_wind_speed": era5_target["wind_speed"],
        "era5_pressure": era5_target["pressure_hpa"],
        "era5_rainfall": era5_target["rainfall_mm"],

        "imd_max_temperature": imd_temp_target["max_temperature"],
        "imd_min_temperature": imd_temp_target["min_temperature"],
        "imd_rainfall": imd_rain_target["rainfall"],
    }
)

# MAHARASHTRA MASK
lat_values = target_lat.values
lon_values = target_lon.values

mask = np.zeros(
    (len(lat_values), len(lon_values)),
    dtype=bool
)

geometry = boundary.geometry.union_all()

for i, lat in enumerate(lat_values):
    for j, lon in enumerate(lon_values):
        mask[i, j] = geometry.contains(Point(float(lon), float(lat)))

master["maharashtra_mask"] = xr.DataArray(
    mask,
    coords={
        "latitude": target_lat,
        "longitude": target_lon,
    },
    dims=("latitude", "longitude")
)

# Apply mask to all climate variables
for var in master.data_vars:
    if var != "maharashtra_mask":
        master[var] = master[var].where(master["maharashtra_mask"])

# METADATA
master.attrs["title"] = (
    "Multi-source Climate Master Dataset for Maharashtra"
)

master.attrs["description"] = (
    "ERA5, IMD and MOSDAC climate variables spatially aligned "
    "to a common Maharashtra grid."
)

master.attrs["sources"] = (
    "ERA5, India Meteorological Department, MOSDAC"
)

master.attrs["temporal_selection"] = (
    "Dates common to ERA5 and IMD rainfall and temperature, "
    "restricted to dates available in the existing MOSDAC dataset."
)

# COVERAGE REPORT
for var in master.data_vars:
    if var == "maharashtra_mask":
        continue

    valid = float(master[var].notnull().mean().values) * 100

    print(f"{var:25s}: {valid:6.2f}%")

# SAVE
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

master.to_netcdf(
    OUTPUT_FILE,
    engine="netcdf4"
)

print("MASTER DATASET REBUILD COMPLETE")
print("\nDimensions:")
print(dict(master.sizes))

print("\nVariables:")
for var in master.data_vars:
    print(" -", var)

print("\nOutput:", OUTPUT_FILE)

# CLOSE
era5.close()
imd_rain.close()
imd_temp.close()
old.close()
master.close()