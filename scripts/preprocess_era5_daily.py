import os
import numpy as np
import xarray as xr

INPUT = r"data\era5\raw\era5_merged_2021_2026.nc"
OUTPUT_DIR = r"data\processed_daily\era5"
OUTPUT = os.path.join(OUTPUT_DIR, "era5_daily.nc")

os.makedirs(OUTPUT_DIR, exist_ok=True)

ds = xr.open_dataset(INPUT)

# Convert units
ds["t2m"] = ds["t2m"] - 273.15
ds["d2m"] = ds["d2m"] - 273.15
ds["sp"] = ds["sp"] / 100.0

# Daily aggregation
daily = xr.Dataset()

daily["temperature_c"] = ds["t2m"].resample(valid_time="1D").mean()
daily["dewpoint_c"] = ds["d2m"].resample(valid_time="1D").mean()

daily["wind_u"] = ds["u10"].resample(valid_time="1D").mean()
daily["wind_v"] = ds["v10"].resample(valid_time="1D").mean()

daily["pressure_hpa"] = ds["sp"].resample(valid_time="1D").mean()

daily["rainfall_mm"] = (
    ds["tp"].resample(valid_time="1D").sum() * 1000
)

# Calculate daily wind speed
daily["wind_speed"] = np.sqrt(
    daily["wind_u"] ** 2 +
    daily["wind_v"] ** 2
)

daily.to_netcdf(OUTPUT)

ds.close()

print("ERA5 daily preprocessing completed.")
print("Output:", OUTPUT)
print("Days:", daily.sizes["valid_time"])