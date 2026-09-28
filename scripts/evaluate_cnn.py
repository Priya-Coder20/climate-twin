import os
import numpy as np
import tensorflow as tf


DATA_DIR = r"data\processed_daily\cnn_dataset\prepared"
MODEL_PATH = r"models\tmax_cnn.keras"


X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
Y_test = np.load(os.path.join(DATA_DIR, "Y_test.npy"))
target_mask = np.load(
    os.path.join(DATA_DIR, "target_mask.npy")
).astype(bool)


X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[2],
    X_test.shape[3],
    X_test.shape[1] * X_test.shape[4]
)


def masked_mse(y_true, y_pred):
    mask = tf.constant(
        target_mask[..., np.newaxis],
        dtype=tf.float32
    )
    error = tf.square(y_true - y_pred) * mask
    return tf.reduce_sum(error) / (
        tf.reduce_sum(mask) + tf.keras.backend.epsilon()
    )


def masked_mae(y_true, y_pred):
    mask = tf.constant(
        target_mask[..., np.newaxis],
        dtype=tf.float32
    )
    error = tf.abs(y_true - y_pred) * mask
    return tf.reduce_sum(error) / (
        tf.reduce_sum(mask) + tf.keras.backend.epsilon()
    )


model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "masked_mse": masked_mse,
        "masked_mae": masked_mae
    }
)


predictions = model.predict(
    X_test,
    batch_size=8,
    verbose=0
)


actual = Y_test[..., np.newaxis]

error = predictions - actual

valid_actual = actual[..., 0][:, target_mask].ravel()
valid_predicted = predictions[..., 0][:, target_mask].ravel()
valid_error = error[..., 0][:, target_mask].ravel()


mae = np.mean(np.abs(valid_error))

rmse = np.sqrt(np.mean(valid_error ** 2))

ss_res = np.sum(
    (valid_actual - valid_predicted) ** 2
)

ss_tot = np.sum(
    (valid_actual - np.mean(valid_actual)) ** 2
)

r2 = 1 - (ss_res / ss_tot)


print("Test evaluation completed.")
print(f"Test MAE:  {mae:.4f} °C")
print(f"Test RMSE: {rmse:.4f} °C")
print(f"Test R²:   {r2:.4f}")