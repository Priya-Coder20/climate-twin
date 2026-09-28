import os
import json
import numpy as np

SEQUENCE_DIR = r"data\processed_daily\cnn_dataset\clean_sequences"
OUTPUT_DIR = r"data\processed_daily\cnn_dataset\clean_prepared"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

os.makedirs(OUTPUT_DIR, exist_ok=True)

X = np.load(os.path.join(SEQUENCE_DIR, "X_temperature.npy"))
Y = np.load(os.path.join(SEQUENCE_DIR, "Y_temperature.npy"))
MASK = np.load(os.path.join(SEQUENCE_DIR, "spatial_mask.npy"))
DATES = np.load(os.path.join(SEQUENCE_DIR, "target_dates.npy"))

print("Loaded:")
print("X:", X.shape)
print("Y:", Y.shape)
print("Mask:", MASK.shape)
print("Dates:", DATES.shape)

n = len(X)

train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

X_train = X[:train_end]
Y_train = Y[:train_end]

X_val = X[train_end:val_end]
Y_val = Y[train_end:val_end]

X_test = X[val_end:]
Y_test = Y[val_end:]

# Keep only valid spatial cells
X_train = X_train[:, :, MASK, :]
X_val = X_val[:, :, MASK, :]
X_test = X_test[:, :, MASK, :]

Y_train = Y_train[:, MASK]
Y_val = Y_val[:, MASK]
Y_test = Y_test[:, MASK]

# Calculate normalization statistics ONLY from training data
feature_mean = np.nanmean(X_train, axis=(0, 1, 2))
feature_std = np.nanstd(X_train, axis=(0, 1, 2))

feature_std[feature_std == 0] = 1.0

# Normalize
X_train = (X_train - feature_mean) / feature_std
X_val = (X_val - feature_mean) / feature_std
X_test = (X_test - feature_mean) / feature_std

# Safety cleanup
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

Y_train = np.nan_to_num(Y_train, nan=0.0)
Y_val = np.nan_to_num(Y_val, nan=0.0)
Y_test = np.nan_to_num(Y_test, nan=0.0)

# Save
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "Y_train.npy"), Y_train)

np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "Y_val.npy"), Y_val)

np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "Y_test.npy"), Y_test)

np.save(os.path.join(OUTPUT_DIR, "target_mask.npy"), MASK)
np.save(os.path.join(OUTPUT_DIR, "feature_mean.npy"), feature_mean)
np.save(os.path.join(OUTPUT_DIR, "feature_std.npy"), feature_std)

np.save(os.path.join(OUTPUT_DIR, "test_dates.npy"), DATES[val_end:])

metadata = {
    "total_sequences": int(n),
    "train_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),
    "input_shape": list(X_train.shape),
    "target_shape": list(Y_train.shape),
    "valid_spatial_cells": int(MASK.sum()),
    "first_date": str(DATES[0]),
    "last_date": str(DATES[-1])
}

with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\nClean data preparation complete.")
print("Train:", X_train.shape, Y_train.shape)
print("Validation:", X_val.shape, Y_val.shape)
print("Test:", X_test.shape, Y_test.shape)
print("Valid spatial cells:", MASK.sum())
print("First date:", DATES[0])
print("Last date:", DATES[-1])