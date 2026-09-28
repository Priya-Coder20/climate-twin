import os
import numpy as np
import xarray as xr

INPUT = r"data\processed_daily\cnn_dataset\common_daily_v2.nc"
OUTPUT_DIR = r"data\processed_daily\cnn_dataset\clean_sequences"

FEATURES = [
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

TARGET = "imd_max_temperature"
WINDOW = 7

os.makedirs(OUTPUT_DIR, exist_ok=True)

ds = xr.open_dataset(INPUT)

dates = ds.time.values.astype("datetime64[D]")

# Load data
data = (
    ds[FEATURES]
    .to_array()
    .transpose(
        "time",
        "latitude",
        "longitude",
        "variable"
    )
    .values
    .astype(np.float32)
)

target = ds[TARGET].values.astype(np.float32)

# Build spatial mask
common_mask = np.ones(
    (len(ds.latitude), len(ds.longitude)),
    dtype=bool
)

for variable in FEATURES:
    common_mask &= np.isfinite(
        ds[variable]
    ).any(dim="time").values

# Apply mask
data[:, ~common_mask, :] = np.nan
target[:, ~common_mask] = np.nan

X = []
Y = []
target_dates = []

rejected_gaps = 0
rejected_missing = 0

for i in range(len(data) - WINDOW):

    window_dates = dates[i:i + WINDOW + 1]

    # Require 8 consecutive calendar days:
    # 7 input days + 1 target day
    diffs = np.diff(window_dates).astype(
        "timedelta64[D]"
    ).astype(int)

    if not np.all(diffs == 1):
        rejected_gaps += 1
        continue

    # Check that the input and target contain valid data
    x_window = data[i:i + WINDOW]
    y_target = target[i + WINDOW]

    if not np.all(
        np.isfinite(
            x_window[:, common_mask, :]
        )
    ):
        rejected_missing += 1
        continue

    if not np.all(
        np.isfinite(
            y_target[common_mask]
        )
    ):
        rejected_missing += 1
        continue

    X.append(x_window)
    Y.append(y_target)
    target_dates.append(dates[i + WINDOW])

X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.float32)
target_dates = np.asarray(
    target_dates,
    dtype="datetime64[D]"
)

np.save(
    os.path.join(OUTPUT_DIR, "X_temperature.npy"),
    X
)

np.save(
    os.path.join(OUTPUT_DIR, "Y_temperature.npy"),
    Y
)

np.save(
    os.path.join(OUTPUT_DIR, "spatial_mask.npy"),
    common_mask
)

np.save(
    os.path.join(OUTPUT_DIR, "target_dates.npy"),
    target_dates
)

print("Clean sequence generation complete.")
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Valid spatial cells:", int(common_mask.sum()))
print("Rejected due to date gaps:", rejected_gaps)
print("Rejected due to missing values:", rejected_missing)

if len(target_dates) > 0:
    print("First target date:", target_dates[0])
    print("Last target date:", target_dates[-1])

ds.close()