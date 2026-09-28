import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PREPARED_DIR = r"data\processed_daily\cnn_dataset\multivariable_prepared"
MODEL_PATH = r"models\temperature_convlstm.keras"
OUTPUT_DIR = r"outputs\temperature_convlstm_evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

X_test = np.load(
    os.path.join(PREPARED_DIR, "X_test.npy")
)

Y_tmax_test = np.load(
    os.path.join(PREPARED_DIR, "Y_tmax_test.npy")
)

Y_tmin_test = np.load(
    os.path.join(PREPARED_DIR, "Y_tmin_test.npy")
)

spatial_mask = np.load(
    os.path.join(PREPARED_DIR, "spatial_mask.npy")
).astype(bool)

# Load model without compiling so custom masked losses are not required.
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

predictions = model.predict(
    X_test,
    verbose=1
)

pred_tmax = predictions[0][..., 0]
pred_tmin = predictions[1][..., 0]

# Target statistics from preprocessing metadata.
tmax_mean = 32.79198455810547
tmax_std = 3.529879093170166

tmin_mean = 20.2198429107666
tmin_std = 3.7381718158721924

# Convert normalized values back to Celsius.
pred_tmax = pred_tmax * tmax_std + tmax_mean
true_tmax = Y_tmax_test * tmax_std + tmax_mean

pred_tmin = pred_tmin * tmin_std + tmin_mean
true_tmin = Y_tmin_test * tmin_std + tmin_mean

# Evaluate only valid Maharashtra grid cells.
true_tmax_valid = true_tmax[:, spatial_mask]
pred_tmax_valid = pred_tmax[:, spatial_mask]

true_tmin_valid = true_tmin[:, spatial_mask]
pred_tmin_valid = pred_tmin[:, spatial_mask]

results = {
    "tmax": {
        "MAE": float(
            mean_absolute_error(
                true_tmax_valid.ravel(),
                pred_tmax_valid.ravel()
            )
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    true_tmax_valid.ravel(),
                    pred_tmax_valid.ravel()
                )
            )
        ),
        "R2": float(
            r2_score(
                true_tmax_valid.ravel(),
                pred_tmax_valid.ravel()
            )
        )
    },
    "tmin": {
        "MAE": float(
            mean_absolute_error(
                true_tmin_valid.ravel(),
                pred_tmin_valid.ravel()
            )
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    true_tmin_valid.ravel(),
                    pred_tmin_valid.ravel()
                )
            )
        ),
        "R2": float(
            r2_score(
                true_tmin_valid.ravel(),
                pred_tmin_valid.ravel()
            )
        )
    }
}

np.save(
    os.path.join(OUTPUT_DIR, "pred_tmax.npy"),
    pred_tmax
)

np.save(
    os.path.join(OUTPUT_DIR, "pred_tmin.npy"),
    pred_tmin
)

with open(
    os.path.join(OUTPUT_DIR, "evaluation_results.txt"),
    "w"
) as f:
    f.write(str(results))

print()
print("=" * 50)
print("TEMPERATURE CONVLSTM EVALUATION")
print("=" * 50)

print("\nTMAX")
print("MAE :", results["tmax"]["MAE"], "°C")
print("RMSE:", results["tmax"]["RMSE"], "°C")
print("R²  :", results["tmax"]["R2"])

print("\nTMIN")
print("MAE :", results["tmin"]["MAE"], "°C")
print("RMSE:", results["tmin"]["RMSE"], "°C")
print("R²  :", results["tmin"]["R2"])

print("\nEvaluation completed.")