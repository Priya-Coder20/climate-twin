import os
import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf


DATA_DIR = r"data\processed_daily\cnn_dataset\prepared"
MODEL_PATH = r"models\tmax_convlstm.keras"
OUTPUT_DIR = r"outputs\convlstm_maps"
COMMON_DATASET = r"data\processed_daily\cnn_dataset\common_daily_v2.nc"

os.makedirs(OUTPUT_DIR, exist_ok=True)


X_test = np.load(
    os.path.join(DATA_DIR, "X_test.npy")
)

Y_test = np.load(
    os.path.join(DATA_DIR, "Y_test.npy")
)

target_mask = np.load(
    os.path.join(DATA_DIR, "target_mask.npy")
).astype(bool)


def masked_mse(y_true, y_pred):

    mask = tf.constant(
        target_mask[..., np.newaxis],
        dtype=tf.float32
    )

    error = tf.square(y_true - y_pred) * mask

    return tf.reduce_sum(error) / (
        tf.reduce_sum(mask) +
        tf.keras.backend.epsilon()
    )


def masked_mae(y_true, y_pred):

    mask = tf.constant(
        target_mask[..., np.newaxis],
        dtype=tf.float32
    )

    error = tf.abs(y_true - y_pred) * mask

    return tf.reduce_sum(error) / (
        tf.reduce_sum(mask) +
        tf.keras.backend.epsilon()
    )


model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "masked_mse": masked_mse,
        "masked_mae": masked_mae
    }
)


sample_index = 0

prediction = model.predict(
    X_test[sample_index:sample_index + 1],
    verbose=0
)[0, ..., 0]

actual = Y_test[sample_index]

error = prediction - actual


prediction[~target_mask] = np.nan
actual[~target_mask] = np.nan
error[~target_mask] = np.nan


ds = xr.open_dataset(COMMON_DATASET)

lat = ds.latitude.values
lon = ds.longitude.values

ds.close()


figures = [
    (actual, "Actual IMD Tmax", "actual_tmax.png"),
    (prediction, "ConvLSTM Predicted Tmax", "predicted_tmax.png"),
    (error, "ConvLSTM Prediction Error", "prediction_error.png")
]


for data, title, filename in figures:

    fig = plt.figure(figsize=(8, 6))

    image = plt.pcolormesh(
        lon,
        lat,
        data,
        shading="auto"
    )

    plt.colorbar(
        image,
        label="Temperature (°C)"
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)

    plt.tight_layout()

    fig.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300,
        format="png"
    )

    plt.close(fig)


print("ConvLSTM maps generated successfully.")