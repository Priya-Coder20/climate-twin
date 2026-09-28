import os
import numpy as np

DATA_DIR = r"data\processed_daily\cnn_dataset\prepared"

X_test = np.load(
    os.path.join(DATA_DIR, "X_test.npy")
)

Y_test = np.load(
    os.path.join(DATA_DIR, "Y_test.npy")
)

target_mask = np.load(
    os.path.join(DATA_DIR, "target_mask.npy")
).astype(bool)

feature_mean = np.load(
    os.path.join(DATA_DIR, "feature_mean.npy")
)

feature_std = np.load(
    os.path.join(DATA_DIR, "feature_std.npy")
)

IMD_MAX_INDEX = 7

today_normalized = X_test[
    :, -1, :, :, IMD_MAX_INDEX
]

today_temperature = (
    today_normalized * feature_std[IMD_MAX_INDEX]
    + feature_mean[IMD_MAX_INDEX]
)

actual = Y_test

valid_actual = actual[:, target_mask]
valid_predicted = today_temperature[:, target_mask]

valid_actual = valid_actual.ravel()
valid_predicted = valid_predicted.ravel()

error = valid_predicted - valid_actual

mae = np.mean(
    np.abs(error)
)

rmse = np.sqrt(
    np.mean(error ** 2)
)

ss_res = np.sum(
    (valid_actual - valid_predicted) ** 2
)

ss_tot = np.sum(
    (valid_actual - np.mean(valid_actual)) ** 2
)

r2 = 1 - (ss_res / ss_tot)

print("Persistence baseline evaluation completed.")
print(f"Baseline MAE:  {mae:.4f} °C")
print(f"Baseline RMSE: {rmse:.4f} °C")
print(f"Baseline R²:   {r2:.4f}")

print(
    f"Baseline temperature range: "
    f"{valid_predicted.min():.2f} to "
    f"{valid_predicted.max():.2f} °C"
)