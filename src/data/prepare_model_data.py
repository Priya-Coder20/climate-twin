import os
import numpy as np
import pandas as pd
import xarray as xr

INPUT_FILE = r"data\processed_daily\cnn_dataset\common_daily_v2.nc"
OUTPUT_DIR = r"data\processed\model_data"

INPUT_DAYS = 7
FORECAST_DAYS = 7

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

TARGETS = [
    "imd_max_temperature",
    "imd_min_temperature",
    "imd_rainfall",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

ds = xr.open_dataset(INPUT_FILE)

# Check variables
missing = [v for v in FEATURES if v not in ds.data_vars]
if missing:
    raise ValueError(f"Missing variables: {missing}")

dates = pd.DatetimeIndex(ds.time.values)

# Find consecutive daily blocks
date_diff = dates.to_series().diff().dt.days
block_id = (date_diff > 1).cumsum()

blocks = []

for _, group in dates.to_series().groupby(block_id):
    if len(group) >= INPUT_DAYS + FORECAST_DAYS:
        blocks.append(np.array([dates.get_loc(d) for d in group.index]))

if not blocks:
    raise ValueError("No consecutive daily blocks are long enough.")

# Load variables
feature_data = np.stack(
    [ds[v].values.astype(np.float32) for v in FEATURES],
    axis=-1
)

target_data = np.stack(
    [ds[v].values.astype(np.float32) for v in TARGETS],
    axis=-1
)

# Find spatial cells that are complete for every variable
# across the entire dataset.
feature_complete = np.isfinite(feature_data).all(axis=(0, 3))
target_complete = np.isfinite(target_data).all(axis=(0, 3))

valid_cells = feature_complete & target_complete

cell_indices = np.argwhere(valid_cells)

if len(cell_indices) == 0:
    raise ValueError("No spatial cells have complete data.")

# Keep only valid spatial cells
feature_data = feature_data[
    :,
    valid_cells,
    :
]

target_data = target_data[
    :,
    valid_cells,
    :
]

X = []
Y = []
sample_dates = []

# Create sequences
for block_indices in blocks:

    for i in range(
        len(block_indices) - INPUT_DAYS - FORECAST_DAYS + 1
    ):

        input_idx = block_indices[
            i:i + INPUT_DAYS
        ]

        target_idx = block_indices[
            i + INPUT_DAYS:
            i + INPUT_DAYS + FORECAST_DAYS
        ]

        x = feature_data[input_idx]
        y = target_data[target_idx]

        if np.isnan(x).any() or np.isnan(y).any():
            continue

        X.append(x)
        Y.append(y)
        sample_dates.append(dates[target_idx[0]])

X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.float32)
sample_dates = np.asarray(sample_dates)

if len(X) == 0:
    raise ValueError("No valid sequences were created.")

# Chronological split
n = len(X)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X[:train_end]
Y_train = Y[:train_end]

X_val = X[train_end:val_end]
Y_val = Y[train_end:val_end]

X_test = X[val_end:]
Y_test = Y[val_end:]

dates_train = sample_dates[:train_end]
dates_val = sample_dates[train_end:val_end]
dates_test = sample_dates[val_end:]

# Save model arrays
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "Y_train.npy"), Y_train)

np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "Y_val.npy"), Y_val)

np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "Y_test.npy"), Y_test)

np.save(os.path.join(OUTPUT_DIR, "dates_train.npy"), dates_train)
np.save(os.path.join(OUTPUT_DIR, "dates_val.npy"), dates_val)
np.save(os.path.join(OUTPUT_DIR, "dates_test.npy"), dates_test)

# Save spatial information for reconstructing maps
np.save(
    os.path.join(OUTPUT_DIR, "valid_cell_indices.npy"),
    cell_indices
)

np.save(
    os.path.join(OUTPUT_DIR, "latitude.npy"),
    ds.latitude.values
)

np.save(
    os.path.join(OUTPUT_DIR, "longitude.npy"),
    ds.longitude.values
)

# Save metadata
with open(os.path.join(OUTPUT_DIR, "metadata.txt"), "w") as f:
    f.write(f"input_days={INPUT_DAYS}\n")
    f.write(f"forecast_days={FORECAST_DAYS}\n")
    f.write(f"features={FEATURES}\n")
    f.write(f"targets={TARGETS}\n")
    f.write(f"valid_spatial_cells={len(cell_indices)}\n")
    f.write(f"samples={n}\n")
    f.write(f"train={len(X_train)}\n")
    f.write(f"validation={len(X_val)}\n")
    f.write(f"test={len(X_test)}\n")

ds.close()

print(f"Created {n} sequences using {len(cell_indices)} spatial cells.")
print(
    f"Train: {len(X_train)}, "
    f"Validation: {len(X_val)}, "
    f"Test: {len(X_test)}"
)