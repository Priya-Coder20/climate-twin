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

# --------------------------------------------------
# 1. Find common spatially valid cells
# --------------------------------------------------

common_mask = np.ones(
    (len(ds.latitude), len(ds.longitude)),
    dtype=bool
)

for variable in features:
    valid = np.isfinite(
        ds[variable]
    ).any(dim="time").values

    common_mask &= valid


# --------------------------------------------------
# 2. Check missing values over time
# --------------------------------------------------

print("Common valid cells:", int(common_mask.sum()))
print()

for variable in features:

    data = ds[variable].values

    # Number of missing values for each common grid cell
    missing_per_cell = np.isnan(data).sum(axis=0)

    values = missing_per_cell[common_mask]

    print(variable)
    print("  Minimum missing days:", int(values.min()))
    print("  Maximum missing days:", int(values.max()))
    print("  Mean missing days:", float(values.mean()))
    print("  Cells with zero missing days:", int((values == 0).sum()))
    print()


ds.close()