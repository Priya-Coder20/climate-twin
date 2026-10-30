import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = r"data\processed\model_data"
MODEL_PATH = r"models\trained\gru_forecaster.keras"

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
Y_test = np.load(os.path.join(DATA_DIR, "Y_test.npy"))

model = tf.keras.models.load_model(MODEL_PATH)

prediction = model.predict(X_test, verbose=0)

target_names = ["tmax", "tmin", "rainfall"]

for i, name in enumerate(target_names):

    actual = Y_test[:, :, :, i].reshape(-1)
    predicted = prediction[:, :, :, i].reshape(-1)

    mask = np.isfinite(actual) & np.isfinite(predicted)

    actual = actual[mask]
    predicted = predicted[mask]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(
        f"{name}: "
        f"MAE={mae:.4f}, "
        f"RMSE={rmse:.4f}, "
        f"R2={r2:.4f}"
    )