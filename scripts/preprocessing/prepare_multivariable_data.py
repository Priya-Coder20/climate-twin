import os
import json
import numpy as np

INPUT_DIR = r"data\processed_daily\cnn_dataset\multivariable_sequences"
OUTPUT_DIR = r"data\processed_daily\cnn_dataset\multivariable_prepared"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

X = np.load(os.path.join(INPUT_DIR, "X_multivariable.npy"))
Y_TMAX = np.load(os.path.join(INPUT_DIR, "Y_tmax.npy"))
Y_TMIN = np.load(os.path.join(INPUT_DIR, "Y_tmin.npy"))
Y_RAIN = np.load(os.path.join(INPUT_DIR, "Y_rainfall.npy"))

feature_names = np.load(
    os.path.join(INPUT_DIR, "feature_names.npy"),
    allow_pickle=True
)

target_dates = np.load(
    os.path.join(INPUT_DIR, "target_dates.npy"),
    allow_pickle=True
)

spatial_mask = np.load(
    os.path.join(INPUT_DIR, "spatial_mask.npy")
).astype(bool)

if X.shape != (760, 7, 65, 73, 10):
    raise ValueError(f"Unexpected X shape: {X.shape}")

if spatial_mask.shape != (65, 73):
    raise ValueError(f"Unexpected spatial mask shape: {spatial_mask.shape}")

if not (
    len(X) == len(Y_TMAX) ==
    len(Y_TMIN) ==
    len(Y_RAIN) ==
    len(target_dates)
):
    raise ValueError("Sequence counts do not match.")

n = len(X)

train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

X_train = X[:train_end]
X_val = X[train_end:val_end]
X_test = X[val_end:]

Y_TMAX_train = Y_TMAX[:train_end]
Y_TMAX_val = Y_TMAX[train_end:val_end]
Y_TMAX_test = Y_TMAX[val_end:]

Y_TMIN_train = Y_TMIN[:train_end]
Y_TMIN_val = Y_TMIN[train_end:val_end]
Y_TMIN_test = Y_TMIN[val_end:]

Y_RAIN_train = Y_RAIN[:train_end]
Y_RAIN_val = Y_RAIN[train_end:val_end]
Y_RAIN_test = Y_RAIN[val_end:]

dates_train = target_dates[:train_end]
dates_val = target_dates[train_end:val_end]
dates_test = target_dates[val_end:]

mask4 = spatial_mask[None, :, :, None]
mask5 = spatial_mask[None, None, :, :, None]

X_train_flat = X_train[:, :, spatial_mask, :]
X_train_flat = X_train_flat.reshape(-1, 10)

feature_mean = np.nanmean(X_train_flat, axis=0)
feature_std = np.nanstd(X_train_flat, axis=0)
feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)

def normalize_x(x):
    result = (
        x - feature_mean.reshape(1, 1, 1, 1, 10)
    ) / feature_std.reshape(1, 1, 1, 1, 10)

    result = np.where(np.isfinite(result), result, 0.0)
    result[:, :, ~spatial_mask, :] = 0.0

    return result.astype(np.float32)

X_train = normalize_x(X_train)
X_val = normalize_x(X_val)
X_test = normalize_x(X_test)

tmax_values = Y_TMAX_train[:, spatial_mask]
tmin_values = Y_TMIN_train[:, spatial_mask]

tmax_mean = np.nanmean(tmax_values)
tmax_std = np.nanstd(tmax_values)
tmax_std = max(tmax_std, 1e-8)

tmin_mean = np.nanmean(tmin_values)
tmin_std = np.nanstd(tmin_values)
tmin_std = max(tmin_std, 1e-8)

def normalize_target(y, mean, std):
    result = (y - mean) / std
    result = np.where(np.isfinite(result), result, 0.0)
    result[:, ~spatial_mask] = 0.0
    return result.astype(np.float32)

Y_TMAX_train = normalize_target(Y_TMAX_train, tmax_mean, tmax_std)
Y_TMAX_val = normalize_target(Y_TMAX_val, tmax_mean, tmax_std)
Y_TMAX_test = normalize_target(Y_TMAX_test, tmax_mean, tmax_std)

Y_TMIN_train = normalize_target(Y_TMIN_train, tmin_mean, tmin_std)
Y_TMIN_val = normalize_target(Y_TMIN_val, tmin_mean, tmin_std)
Y_TMIN_test = normalize_target(Y_TMIN_test, tmin_mean, tmin_std)

rain_train_log = np.log1p(
    np.maximum(Y_RAIN_train[:, spatial_mask], 0)
)

rain_mean = np.nanmean(rain_train_log)
rain_std = np.nanstd(rain_train_log)
rain_std = max(rain_std, 1e-8)

def normalize_rainfall(y):
    result = np.zeros_like(y, dtype=np.float32)

    valid = np.isfinite(y)
    transformed = np.zeros_like(y, dtype=np.float32)

    transformed[valid] = np.log1p(
        np.maximum(y[valid], 0)
    )

    result[valid] = (
        (transformed[valid] - rain_mean) / rain_std
    )

    result[:, ~spatial_mask] = 0.0

    return result

Y_RAIN_train = normalize_rainfall(Y_RAIN_train)
Y_RAIN_val = normalize_rainfall(Y_RAIN_val)
Y_RAIN_test = normalize_rainfall(Y_RAIN_test)

arrays = {
    "X_train": X_train,
    "X_val": X_val,
    "X_test": X_test,
    "Y_tmax_train": Y_TMAX_train,
    "Y_tmax_val": Y_TMAX_val,
    "Y_tmax_test": Y_TMAX_test,
    "Y_tmin_train": Y_TMIN_train,
    "Y_tmin_val": Y_TMIN_val,
    "Y_tmin_test": Y_TMIN_test,
    "Y_rainfall_train": Y_RAIN_train,
    "Y_rainfall_val": Y_RAIN_val,
    "Y_rainfall_test": Y_RAIN_test
}

for name, arr in arrays.items():
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf values.")

    np.save(
        os.path.join(OUTPUT_DIR, f"{name}.npy"),
        arr
    )

np.save(
    os.path.join(OUTPUT_DIR, "feature_mean.npy"),
    feature_mean.astype(np.float32)
)

np.save(
    os.path.join(OUTPUT_DIR, "feature_std.npy"),
    feature_std.astype(np.float32)
)

np.save(
    os.path.join(OUTPUT_DIR, "target_dates_train.npy"),
    dates_train
)

np.save(
    os.path.join(OUTPUT_DIR, "target_dates_val.npy"),
    dates_val
)

np.save(
    os.path.join(OUTPUT_DIR, "target_dates_test.npy"),
    dates_test
)

np.save(
    os.path.join(OUTPUT_DIR, "spatial_mask.npy"),
    spatial_mask
)

metadata = {
    "total_sequences": n,
    "train_sequences": len(X_train),
    "validation_sequences": len(X_val),
    "test_sequences": len(X_test),
    "input_shape": [7, 65, 73, 10],
    "features": [str(x) for x in feature_names],
    "targets": ["tmax", "tmin", "rainfall"],
    "spatial_cells": int(spatial_mask.sum()),
    "split": "chronological",
    "rainfall_transform": "log1p",
    "target_statistics": {
        "tmax_mean": float(tmax_mean),
        "tmax_std": float(tmax_std),
        "tmin_mean": float(tmin_mean),
        "tmin_std": float(tmin_std),
        "rainfall_log_mean": float(rain_mean),
        "rainfall_log_std": float(rain_std)
    }
}

with open(
    os.path.join(OUTPUT_DIR, "metadata.json"),
    "w"
) as f:
    json.dump(metadata, f, indent=2)

print("Multivariable preprocessing completed.")