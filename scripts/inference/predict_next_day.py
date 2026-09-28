import os
import json
import numpy as np
import tensorflow as tf

PREPARED_DIR = r"data\processed_daily\cnn_dataset\multivariable_prepared"
MODEL_PATH = r"models\temperature_convlstm.keras"
OUTPUT_DIR = r"outputs\next_day_prediction"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load prepared test sequence
X_test = np.load(
    os.path.join(PREPARED_DIR, "X_test.npy")
)

target_dates = np.load(
    os.path.join(PREPARED_DIR, "target_dates_test.npy")
)

spatial_mask = np.load(
    os.path.join(PREPARED_DIR, "spatial_mask.npy")
).astype(bool)

# Load target statistics used during preprocessing
with open(
    os.path.join(PREPARED_DIR, "metadata.json"),
    "r"
) as f:
    metadata = json.load(f)

tmax_mean = metadata["target_statistics"]["tmax_mean"]
tmax_std = metadata["target_statistics"]["tmax_std"]

tmin_mean = metadata["target_statistics"]["tmin_mean"]
tmin_std = metadata["target_statistics"]["tmin_std"]

# Load final trained temperature model
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# Use the last available historical 7-day sequence
X_input = X_test[-1:]

prediction_date = str(target_dates[-1])

# Predict Tmax and Tmin
predictions = model.predict(
    X_input,
    verbose=1
)

pred_tmax = predictions[0][0, ..., 0]
pred_tmin = predictions[1][0, ..., 0]

# Convert back to Celsius
pred_tmax = (
    pred_tmax * tmax_std
    + tmax_mean
)

pred_tmin = (
    pred_tmin * tmin_std
    + tmin_mean
)

# Apply Maharashtra spatial mask
pred_tmax[~spatial_mask] = np.nan
pred_tmin[~spatial_mask] = np.nan

# Save prediction grids
np.save(
    os.path.join(
        OUTPUT_DIR,
        "predicted_tmax.npy"
    ),
    pred_tmax
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "predicted_tmin.npy"
    ),
    pred_tmin
)

# Save metadata
prediction_info = {
    "prediction_date": prediction_date,
    "model": "temperature_convlstm.keras",
    "input_shape": list(X_input.shape),
    "spatial_cells": int(spatial_mask.sum()),
    "tmax_min_celsius": float(np.nanmin(pred_tmax)),
    "tmax_max_celsius": float(np.nanmax(pred_tmax)),
    "tmax_mean_celsius": float(np.nanmean(pred_tmax)),
    "tmin_min_celsius": float(np.nanmin(pred_tmin)),
    "tmin_max_celsius": float(np.nanmax(pred_tmin)),
    "tmin_mean_celsius": float(np.nanmean(pred_tmin))
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "prediction_metadata.json"
    ),
    "w"
) as f:
    json.dump(
        prediction_info,
        f,
        indent=2
    )

print()
print("=" * 50)
print("NEXT-DAY TEMPERATURE INFERENCE")
print("=" * 50)

print("Prediction date:", prediction_date)

print()
print("TMAX")
print("Minimum:", prediction_info["tmax_min_celsius"], "°C")
print("Maximum:", prediction_info["tmax_max_celsius"], "°C")
print("Mean   :", prediction_info["tmax_mean_celsius"], "°C")

print()
print("TMIN")
print("Minimum:", prediction_info["tmin_min_celsius"], "°C")
print("Maximum:", prediction_info["tmin_max_celsius"], "°C")
print("Mean   :", prediction_info["tmin_mean_celsius"], "°C")

print()
print("Saved:")
print(
    os.path.join(
        OUTPUT_DIR,
        "predicted_tmax.npy"
    )
)
print(
    os.path.join(
        OUTPUT_DIR,
        "predicted_tmin.npy"
    )
)

print()
print("Inference completed.")