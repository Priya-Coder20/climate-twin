import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PREPARED_DIR = r"data\processed_daily\cnn_dataset\multivariable_prepared"
MODEL_PATH = r"models\multivariable_convlstm.keras"
OUTPUT_DIR = r"outputs\multivariable_convlstm_evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

X_test = np.load(os.path.join(PREPARED_DIR, "X_test.npy"))
Y_tmax = np.load(os.path.join(PREPARED_DIR, "Y_tmax_test.npy"))
Y_tmin = np.load(os.path.join(PREPARED_DIR, "Y_tmin_test.npy"))
Y_rain = np.load(os.path.join(PREPARED_DIR, "Y_rainfall_test.npy"))

with open(os.path.join(PREPARED_DIR, "metadata.json"), "r") as f:
    metadata = json.load(f)

mask = np.load(os.path.join(PREPARED_DIR, "spatial_mask.npy"))

stats = metadata["target_statistics"]

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

predictions = model.predict(X_test, verbose=1)

if isinstance(predictions, dict):
    pred_tmax = predictions["tmax_output"]
    pred_tmin = predictions["tmin_output"]
    pred_rain = predictions["rainfall_output"]
else:
    pred_tmax, pred_tmin, pred_rain = predictions

valid = mask.astype(bool)

tmax_mean = stats["tmax_mean"]
tmax_std = stats["tmax_std"]

tmin_mean = stats["tmin_mean"]
tmin_std = stats["tmin_std"]

rain_mean = stats["rainfall_log_mean"]
rain_std = stats["rainfall_log_std"]

pred_tmax = pred_tmax * tmax_std + tmax_mean
Y_tmax = Y_tmax * tmax_std + tmax_mean

pred_tmin = pred_tmin * tmin_std + tmin_mean
Y_tmin = Y_tmin * tmin_std + tmin_mean

pred_rain_log = pred_rain * rain_std + rain_mean
true_rain_log = Y_rain * rain_std + rain_mean

pred_rain = np.expm1(pred_rain_log)
Y_rain = np.expm1(true_rain_log)

pred_rain = np.maximum(pred_rain, 0)

results = {}

for name, true, pred in [
    ("tmax", Y_tmax, pred_tmax),
    ("tmin", Y_tmin, pred_tmin),
    ("rainfall", Y_rain, pred_rain),
]:
    y_true = true[:, valid].reshape(-1)
    y_pred = pred[:, valid].reshape(-1)

    results[name] = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError(f"Invalid values detected in {name} evaluation.")

with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), "w") as f:
    json.dump(results, f, indent=2)

np.save(os.path.join(OUTPUT_DIR, "pred_tmax.npy"), pred_tmax)
np.save(os.path.join(OUTPUT_DIR, "pred_tmin.npy"), pred_tmin)
np.save(os.path.join(OUTPUT_DIR, "pred_rainfall.npy"), pred_rain)

print(json.dumps(results, indent=2))
print("Multivariable ConvLSTM evaluation completed.")