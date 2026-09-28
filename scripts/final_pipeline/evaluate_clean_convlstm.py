import os
import numpy as np
import tensorflow as tf

MODEL_PATH = r"models\tmax_convlstm_clean.keras"
DATA_DIR = r"data\processed_daily\cnn_dataset\clean_prepared"
OUTPUT_DIR = r"outputs\convlstm_clean"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
Y_test = np.load(os.path.join(DATA_DIR, "Y_test.npy"))
target_mask = np.load(os.path.join(DATA_DIR, "target_mask.npy"))
feature_mean = np.load(os.path.join(DATA_DIR, "feature_mean.npy"))
feature_std = np.load(os.path.join(DATA_DIR, "feature_std.npy"))
test_dates = np.load(os.path.join(DATA_DIR, "test_dates.npy"))


model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False,
    custom_objects={"tf": tf}
)


# Restore sparse data to spatial grid

valid_indices = np.where(target_mask.ravel())[0]

n = len(X_test)
h, w = target_mask.shape

X_full = np.zeros((n, 7, h, w, X_test.shape[-1]), dtype=np.float32)
Y_full = np.zeros((n, h, w), dtype=np.float32)

for i in range(n):
    for t in range(7):
        X_full[i, t].reshape(-1, X_test.shape[-1])[valid_indices] = X_test[i, t]

    Y_full[i].ravel()[valid_indices] = Y_test[i]


# ConvLSTM prediction


pred = model.predict(
    X_full,
    verbose=0
)

pred = np.asarray(pred)

if pred.ndim == 4:
    pred = np.squeeze(pred, axis=-1)


# Evaluate only valid Maharashtra cells

mask = target_mask.astype(bool)

y_true = Y_full[:, mask].reshape(-1)
y_pred = pred[:, mask].reshape(-1)

mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

r2 = 1 - (ss_res / ss_tot)


# Persistence baseline
# Previous day's IMD Tmax
# Feature index 7 = imd_max_temperature


prev_tmax = X_full[:, -1, :, :, 7]

prev_tmax = (
    prev_tmax * feature_std[7]
    + feature_mean[7]
)

baseline_true = Y_full[:, mask].reshape(-1)
baseline_pred = prev_tmax[:, mask].reshape(-1)

baseline_mae = np.mean(
    np.abs(baseline_true - baseline_pred)
)

baseline_rmse = np.sqrt(
    np.mean((baseline_true - baseline_pred) ** 2)
)

baseline_ss_res = np.sum(
    (baseline_true - baseline_pred) ** 2
)

baseline_ss_tot = np.sum(
    (baseline_true - np.mean(baseline_true)) ** 2
)

baseline_r2 = 1 - (
    baseline_ss_res / baseline_ss_tot
)


# Save predictions

np.save(
    os.path.join(OUTPUT_DIR, "test_predictions.npy"),
    pred
)

np.save(
    os.path.join(OUTPUT_DIR, "test_actual.npy"),
    Y_full
)

np.save(
    os.path.join(OUTPUT_DIR, "test_dates.npy"),
    test_dates
)


# Results
print(f"Test samples: {len(X_test)}")
print(f"Valid spatial cells: {mask.sum()}")

print(f"MAE:  {mae:.4f} °C")
print(f"RMSE: {rmse:.4f} °C")
print(f"R²:   {r2:.4f}")

print()
print(f"MAE:  {baseline_mae:.4f} °C")
print(f"RMSE: {baseline_rmse:.4f} °C")
print(f"R²:   {baseline_r2:.4f}")
