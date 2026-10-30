import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = r"data\processed\convlstm_data"
MODEL_PATH = r"models\trained\convlstm_forecaster.keras"

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
Y_test = np.load(os.path.join(DATA_DIR, "Y_test.npy"))

target_mean = np.load(
    os.path.join(DATA_DIR, "target_mean.npy")
)

target_std = np.load(
    os.path.join(DATA_DIR, "target_std.npy")
)

usable_cells = np.load(
    os.path.join(DATA_DIR, "usable_cells.npy")
)

model = tf.keras.models.load_model(MODEL_PATH)

prediction = model.predict(
    X_test,
    verbose=0
)

prediction = prediction * target_std + target_mean
actual = Y_test * target_std + target_mean

prediction = prediction[:, :, usable_cells, :]
actual = actual[:, :, usable_cells, :]

target_names = [
    "tmax",
    "tmin",
    "rainfall"
]

for day in range(7):

    print(f"\nDay {day + 1}")

    for i, name in enumerate(target_names):

        y_true = actual[:, day, :, i].reshape(-1)
        y_pred = prediction[:, day, :, i].reshape(-1)

        mask = np.isfinite(y_true) & np.isfinite(y_pred)

        y_true = y_true[mask]
        y_pred = y_pred[mask]

        mae = mean_absolute_error(
            y_true,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )

        r2 = r2_score(
            y_true,
            y_pred
        )

        print(
            f"{name}: "
            f"MAE={mae:.4f}, "
            f"RMSE={rmse:.4f}, "
            f"R2={r2:.4f}"
        )