import os
import json
import numpy as np
import tensorflow as tf

PREPARED_DIR = r"data\processed_daily\cnn_dataset\multivariable_prepared"
MODEL_PATH = r"models\multivariable_convlstm.keras"

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

with open(os.path.join(PREPARED_DIR, "metadata.json"), "r") as f:
    metadata = json.load(f)

mask = np.load(
    os.path.join(PREPARED_DIR, "spatial_mask.npy")
).astype(bool)

stats = metadata["target_statistics"]

sets = {
    "train": (
        "X_train.npy",
        "Y_tmax_train.npy"
    ),
    "val": (
        "X_val.npy",
        "Y_tmax_val.npy"
    ),
    "test": (
        "X_test.npy",
        "Y_tmax_test.npy"
    ),
}

mean = stats["tmax_mean"]
std = stats["tmax_std"]

for name, (x_file, y_file) in sets.items():

    X = np.load(os.path.join(PREPARED_DIR, x_file))
    Y = np.load(os.path.join(PREPARED_DIR, y_file))

    predictions = model.predict(X, verbose=0)

    if isinstance(predictions, dict):
        pred = predictions["tmax_output"]
    else:
        pred = predictions[0]

    pred = pred * std + mean
    Y = Y * std + mean

    y_true = Y[:, mask].reshape(-1)
    y_pred = pred[:, mask].reshape(-1)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print()
    print(name.upper())
    print("MAE :", round(float(mae), 4))
    print("RMSE:", round(float(rmse), 4))
    print("R2  :", round(float(r2), 4))
    print("True mean:", round(float(y_true.mean()), 4))
    print("Pred mean:", round(float(y_pred.mean()), 4))
    print("True std :", round(float(y_true.std()), 4))
    print("Pred std :", round(float(y_pred.std()), 4))