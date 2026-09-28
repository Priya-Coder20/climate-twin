import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


DATA_DIR = r"data\processed_daily\cnn_dataset\prepared"
MODEL_DIR = r"models"

os.makedirs(MODEL_DIR, exist_ok=True)


X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))

X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
Y_val = np.load(os.path.join(DATA_DIR, "Y_val.npy"))

target_mask = np.load(
    os.path.join(DATA_DIR, "target_mask.npy")
).astype(np.float32)


X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[2],
    X_train.shape[3],
    X_train.shape[1] * X_train.shape[4]
)

X_val = X_val.reshape(
    X_val.shape[0],
    X_val.shape[2],
    X_val.shape[3],
    X_val.shape[1] * X_val.shape[4]
)


mask = tf.constant(target_mask[..., np.newaxis], dtype=tf.float32)


def masked_mse(y_true, y_pred):
    error = tf.square(y_true - y_pred)
    error = error * mask

    return tf.reduce_sum(error) / (
        tf.reduce_sum(mask) + tf.keras.backend.epsilon()
    )


def masked_mae(y_true, y_pred):
    error = tf.abs(y_true - y_pred)
    error = error * mask

    return tf.reduce_sum(error) / (
        tf.reduce_sum(mask) + tf.keras.backend.epsilon()
    )


inputs = layers.Input(shape=(65, 73, 70))

x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = layers.BatchNormalization()(x)

x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)

outputs = layers.Conv2D(1, 1, padding="same")(x)

model = models.Model(inputs, outputs)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=masked_mse,
    metrics=[masked_mae]
)


model.summary()


checkpoint = callbacks.ModelCheckpoint(
    os.path.join(MODEL_DIR, "tmax_cnn.keras"),
    monitor="val_loss",
    save_best_only=True
)

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


history = model.fit(
    X_train,
    Y_train[..., np.newaxis],
    validation_data=(
        X_val,
        Y_val[..., np.newaxis]
    ),
    epochs=50,
    batch_size=8,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)


np.save(
    os.path.join(MODEL_DIR, "training_loss.npy"),
    np.array(history.history["loss"])
)

np.save(
    os.path.join(MODEL_DIR, "validation_loss.npy"),
    np.array(history.history["val_loss"])
)

print("Training completed.")
print(f"Model saved to: {os.path.join(MODEL_DIR, 'tmax_cnn.keras')}")