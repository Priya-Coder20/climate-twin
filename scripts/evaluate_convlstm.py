import os
import numpy as np
import tensorflow as tf


DATA_DIR = r"data\processed_daily\cnn_dataset\prepared"
MODEL_PATH = r"models\tmax_convlstm.keras"


X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
Y_test = np.load(os.path.join(DATA_DIR, "Y_test.npy"))
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
    batch_size=4,
    verbose=0
)


actual = Y_test[..., np.newaxis]

valid_actual = actual[..., 0][:, target_mask].ravel()
valid_predicted = predictions[..., 0][:, target_mask].ravel()

error = valid_predicted - valid_actual


mae = np.mean(np.abs(error))

rmse = np.sqrt(np.mean(error ** 2))

ss_res = np.sum(error ** 2)

ss_tot = np.sum(
    (valid_actual - np.mean(valid_actual)) ** 2
)

r2 = 1 - (ss_res / ss_tot)


print("ConvLSTM test evaluation completed.")
print(f"Test MAE:  {mae:.4f} °C")
print(f"Test RMSE: {rmse:.4f} °C")
print(f"Test R²:   {r2:.4f}")