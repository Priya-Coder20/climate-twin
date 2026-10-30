import os
import numpy as np
import xarray as xr

DATA_PATH = r"data\processed\master_climate_dataset.nc"
OUTPUT_DIR = r"data\processed\convlstm_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    "imd_rainfall"
]

TARGETS = [
    "imd_max_temperature",
    "imd_min_temperature",
    "imd_rainfall"
]

INPUT_DAYS = 7
FORECAST_DAYS = 7
MIN_COVERAGE = 0.80

ds = xr.open_dataset(DATA_PATH)

mask = ds["maharashtra_mask"].values.astype(bool)

features = np.stack(
    [ds[v].values for v in FEATURES],
    axis=-1
)

targets = np.stack(
    [ds[v].values for v in TARGETS],
    axis=-1
)

dates = ds.time.values

feature_valid = np.isfinite(features).all(axis=-1)
target_valid = np.isfinite(targets).all(axis=-1)

combined_valid = feature_valid & target_valid

coverage = combined_valid.mean(axis=0)

usable_cells = mask & (coverage >= MIN_COVERAGE)

X = []
Y = []

for i in range(
    len(dates) - INPUT_DAYS - FORECAST_DAYS + 1
):

    window_dates = dates[
        i:i + INPUT_DAYS + FORECAST_DAYS
    ]

    if not np.all(
        np.diff(window_dates) == np.timedelta64(1, "D")
    ):
        continue

    x = features[
        i:i + INPUT_DAYS
    ].copy()

    y = targets[
        i + INPUT_DAYS:
        i + INPUT_DAYS + FORECAST_DAYS
    ].copy()

    if not np.isfinite(
        x[:, usable_cells, :]
    ).all():
        continue

    if not np.isfinite(
        y[:, usable_cells, :]
    ).all():
        continue

    x[:, ~mask, :] = 0
    y[:, ~mask, :] = 0

    X.append(x)
    Y.append(y)

if not X:
    raise RuntimeError(
        "No valid ConvLSTM sequences were created."
    )

X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.float32)

n = len(X)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X[:train_end]
Y_train = Y[:train_end]

X_val = X[train_end:val_end]
Y_val = Y[train_end:val_end]

X_test = X[val_end:]
Y_test = Y[val_end:]

train_values = X_train[:, :, usable_cells, :].reshape(
    -1, len(FEATURES)
)

feature_mean = np.nanmean(train_values, axis=0)
feature_std = np.nanstd(train_values, axis=0)

target_values = Y_train[:, :, usable_cells, :].reshape(
    -1, len(TARGETS)
)

target_mean = np.nanmean(target_values, axis=0)
target_std = np.nanstd(target_values, axis=0)

feature_std[feature_std == 0] = 1
target_std[target_std == 0] = 1

X_train = (X_train - feature_mean) / feature_std
X_val = (X_val - feature_mean) / feature_std
X_test = (X_test - feature_mean) / feature_std

Y_train = (Y_train - target_mean) / target_std
Y_val = (Y_val - target_mean) / target_std
Y_test = (Y_test - target_mean) / target_std

# Ensure TensorFlow receives only finite values.
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

Y_train = np.nan_to_num(Y_train, nan=0.0, posinf=0.0, neginf=0.0)
Y_val = np.nan_to_num(Y_val, nan=0.0, posinf=0.0, neginf=0.0)
Y_test = np.nan_to_num(Y_test, nan=0.0, posinf=0.0, neginf=0.0)

np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "Y_train.npy"), Y_train)

np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "Y_val.npy"), Y_val)

np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "Y_test.npy"), Y_test)

np.save(
    os.path.join(OUTPUT_DIR, "feature_mean.npy"),
    feature_mean
)

np.save(
    os.path.join(OUTPUT_DIR, "feature_std.npy"),
    feature_std
)

np.save(
    os.path.join(OUTPUT_DIR, "target_mean.npy"),
    target_mean
)

np.save(
    os.path.join(OUTPUT_DIR, "target_std.npy"),
    target_std
)

np.save(
    os.path.join(OUTPUT_DIR, "maharashtra_mask.npy"),
    mask
)

np.save(
    os.path.join(OUTPUT_DIR, "usable_cells.npy"),
    usable_cells
)

ds.close()

print(X_train.shape, X_val.shape, X_test.shape)