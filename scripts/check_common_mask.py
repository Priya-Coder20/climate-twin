import numpy as np
import xarray as xr

INPUT = r"data\processed_daily\cnn_dataset\common_daily_v2.nc"

features = [
    "imr",
    "pet",
    "era5_temperature",
    "era5_dewpoint",
    "era5_wind_speed",
    "era5_pressure",
    "era5_rainfall",
    "imd_max_temperature",
    "imd_min_temperature",
    "imd_rainfall",
]

ds = xr.open_dataset(INPUT)

common = np.ones(
    (len(ds.latitude), len(ds.longitude)),
    dtype=bool
)

for variable in features:
    valid = np.isfinite(ds[variable]).any(dim="time").values
    common &= valid

print("Common valid cells:", int(common.sum()))
print("Common permanently invalid cells:", int((~common).sum()))
print("Total cells:", common.size)

ds.close()