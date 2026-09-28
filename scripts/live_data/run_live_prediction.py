import os
import json
import numpy as np
import tensorflow as tf

BASE_DIR = r"C:\Users\HP\climate-twin"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "live",
    "model_input",
    "X_live.npy"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "temperature_convlstm.keras"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "live_prediction"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TMAX_MEAN = 32.7919846
TMAX_STD = 3.5298791

TMIN_MEAN = 19.755241
TMIN_STD = 0.8736007

INPUT_DATES = [
    "2026-09-14",
    "2026-09-15",
    "2026-09-16",
    "2026-09-17",
    "2026-09-18",
    "2026-09-19",
    "2026-09-20"
]

PREDICTION_DATE = "2026-09-21"

X = np.load(INPUT_FILE).astype(np.float32)

if X.shape != (7, 65, 73, 10):
    raise ValueError(
        f"Unexpected X shape: {X.shape}"
    )

X = np.expand_dims(X, axis=0)

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_FILE,
    compile=False
)

print("Input shape:", X.shape)

prediction = model.predict(
    X,
    verbose=1
)

if isinstance(prediction, (list, tuple)):

    tmax_pred = np.asarray(
        prediction[0][0],
        dtype=np.float32
    )

    tmin_pred = np.asarray(
        prediction[1][0],
        dtype=np.float32
    )

else:

    prediction = np.asarray(
        prediction,
        dtype=np.float32
    )

    if prediction.shape[-1] != 2:
        raise ValueError(
            f"Unexpected prediction shape: {prediction.shape}"
        )

    tmax_pred = prediction[0, :, :, 0]
    tmin_pred = prediction[0, :, :, 1]

print("\nNormalized model output:")

print(
    "Tmax:",
    f"min={tmax_pred.min():.3f}",
    f"max={tmax_pred.max():.3f}",
    f"mean={tmax_pred.mean():.3f}"
)

print(
    "Tmin:",
    f"min={tmin_pred.min():.3f}",
    f"max={tmin_pred.max():.3f}",
    f"mean={tmin_pred.mean():.3f}"
)

tmax_pred = (
    tmax_pred * TMAX_STD
    + TMAX_MEAN
)

tmin_pred = (
    tmin_pred * TMIN_STD
    + TMIN_MEAN
)

tmax_pred = tmax_pred.astype(np.float32)
tmin_pred = tmin_pred.astype(np.float32)

tmax_file = os.path.join(
    OUTPUT_DIR,
    "live_tmax.npy"
)

tmin_file = os.path.join(
    OUTPUT_DIR,
    "live_tmin.npy"
)

np.save(
    tmax_file,
    tmax_pred
)

np.save(
    tmin_file,
    tmin_pred
)

metadata = {
    "input_dates": INPUT_DATES,
    "prediction_date": PREDICTION_DATE,
    "model": "temperature_convlstm.keras",
    "input_shape": [1, 7, 65, 73, 10],
    "tmax_shape": list(tmax_pred.shape),
    "tmin_shape": list(tmin_pred.shape),
    "units": "degree Celsius",
    "inverse_normalization": {
        "tmax_mean": TMAX_MEAN,
        "tmax_std": TMAX_STD,
        "tmin_mean": TMIN_MEAN,
        "tmin_std": TMIN_STD
    },
    "tmax_min": float(tmax_pred.min()),
    "tmax_max": float(tmax_pred.max()),
    "tmax_mean": float(tmax_pred.mean()),
    "tmin_min": float(tmin_pred.min()),
    "tmin_max": float(tmin_pred.max()),
    "tmin_mean": float(tmin_pred.mean())
}

metadata_file = os.path.join(
    OUTPUT_DIR,
    "prediction_metadata.json"
)

with open(
    metadata_file,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )

print("\n============================================================")
print("LIVE PREDICTION COMPLETED")
print("============================================================")

print(
    "Prediction date:",
    PREDICTION_DATE
)

print(
    "Tmax:",
    f"min={tmax_pred.min():.3f}",
    f"max={tmax_pred.max():.3f}",
    f"mean={tmax_pred.mean():.3f}",
    "°C"
)

print(
    "Tmin:",
    f"min={tmin_pred.min():.3f}",
    f"max={tmin_pred.max():.3f}",
    f"mean={tmin_pred.mean():.3f}",
    "°C"
)

print(
    "Tmax saved:",
    tmax_file
)

print(
    "Tmin saved:",
    tmin_file
)

print(
    "Metadata saved:",
    metadata_file
)