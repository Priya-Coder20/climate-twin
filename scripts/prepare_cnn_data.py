import os
import json
import numpy as np


SEQUENCE_DIR = r"data\processed_daily\cnn_dataset\sequences"
OUTPUT_DIR = r"data\processed_daily\cnn_dataset\prepared"

X_FILE = os.path.join(SEQUENCE_DIR, "X_temperature.npy")
Y_FILE = os.path.join(SEQUENCE_DIR, "Y_temperature.npy")
MASK_FILE = os.path.join(SEQUENCE_DIR, "spatial_mask.npy")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

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

os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_values(arr, feature_name):
    arr = arr.astype(np.float32, copy=True)
    arr[~np.isfinite(arr)] = np.nan

    if feature_name in ["imr", "pet"]:
        arr[arr <= -900] = np.nan

    return arr


def calculate_statistics(values):
    valid = values[np.isfinite(values)]

    if valid.size == 0:
        raise ValueError("No valid values available for normalization.")

    mean = np.mean(valid)
    std = np.std(valid)

    if std < 1e-8:
        std = 1.0

    return float(mean), float(std)


X = np.load(X_FILE).astype(np.float32)
Y = np.load(Y_FILE).astype(np.float32)
spatial_mask = np.load(MASK_FILE).astype(bool)


if X.ndim != 5:
    raise ValueError(f"Unexpected X shape: {X.shape}")

if Y.ndim != 3:
    raise ValueError(f"Unexpected Y shape: {Y.shape}")

n_samples, n_days, n_lat, n_lon, n_features = X.shape

if n_features != len(FEATURES):
    raise ValueError(
        f"Expected {len(FEATURES)} features, found {n_features}"
    )

if Y.shape != (n_samples, n_lat, n_lon):
    raise ValueError(f"Unexpected Y shape: {Y.shape}")

if spatial_mask.shape != (n_lat, n_lon):
    raise ValueError(
        f"Unexpected spatial mask shape: {spatial_mask.shape}"
    )


train_end = int(n_samples * TRAIN_RATIO)
val_end = train_end + int(n_samples * VAL_RATIO)

train_indices = np.arange(0, train_end)
val_indices = np.arange(train_end, val_end)
test_indices = np.arange(val_end, n_samples)


for i, feature in enumerate(FEATURES):
    X[..., i] = clean_values(X[..., i], feature)

Y[~np.isfinite(Y)] = np.nan


for i, feature in enumerate(FEATURES):
    X[..., i][..., ~spatial_mask] = np.nan

Y[..., ~spatial_mask] = np.nan


feature_means = np.zeros(n_features, dtype=np.float32)
feature_stds = np.ones(n_features, dtype=np.float32)

for i, feature in enumerate(FEATURES):
    train_values = X[train_indices, ..., i]

    mean, std = calculate_statistics(train_values)

    feature_means[i] = mean
    feature_stds[i] = std

    X[..., i] = (X[..., i] - mean) / std


X = np.nan_to_num(
    X,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

Y = np.nan_to_num(
    Y,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


X_train = X[train_indices]
Y_train = Y[train_indices]

X_val = X[val_indices]
Y_val = Y[val_indices]

X_test = X[test_indices]
Y_test = Y[test_indices]


np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "Y_train.npy"), Y_train)

np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "Y_val.npy"), Y_val)

np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "Y_test.npy"), Y_test)

np.save(os.path.join(OUTPUT_DIR, "target_mask.npy"), spatial_mask)

np.save(
    os.path.join(OUTPUT_DIR, "feature_mean.npy"),
    feature_means
)

np.save(
    os.path.join(OUTPUT_DIR, "feature_std.npy"),
    feature_stds
)


metadata = {
    "features": FEATURES,
    "target": TARGET,
    "input_shape": list(X.shape),
    "target_shape": list(Y.shape),
    "train_samples": int(len(train_indices)),
    "validation_samples": int(len(val_indices)),
    "test_samples": int(len(test_indices)),
    "train_ratio": TRAIN_RATIO,
    "validation_ratio": VAL_RATIO,
    "test_ratio": TEST_RATIO,
    "latitude_cells": int(n_lat),
    "longitude_cells": int(n_lon),
    "valid_target_cells": int(spatial_mask.sum()),
    "invalid_target_cells": int(spatial_mask.size - spatial_mask.sum()),
    "feature_means": feature_means.tolist(),
    "feature_stds": feature_stds.tolist(),
    "target_normalized": False
}

with open(
    os.path.join(OUTPUT_DIR, "metadata.json"),
    "w"
) as f:
    json.dump(metadata, f, indent=4)


print(f"X_train: {X_train.shape}")
print(f"X_val:   {X_val.shape}")
print(f"X_test:  {X_test.shape}")
print(f"Y_train: {Y_train.shape}")
print(f"Y_val:   {Y_val.shape}")
print(f"Y_test:  {Y_test.shape}")
print(f"Valid spatial cells: {int(spatial_mask.sum())}")
print(f"Invalid spatial cells: {int((~spatial_mask).sum())}")