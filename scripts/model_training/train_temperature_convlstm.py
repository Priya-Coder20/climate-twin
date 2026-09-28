import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

PREPARED_DIR = r"data\processed_daily\cnn_dataset\multivariable_prepared"
MODEL_DIR = r"models"
OUTPUT_DIR = r"outputs\temperature_convlstm"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_train = np.load(
    os.path.join(PREPARED_DIR, "X_train.npy")
)

X_val = np.load(
    os.path.join(PREPARED_DIR, "X_val.npy")
)

Y_tmax_train = np.load(
    os.path.join(PREPARED_DIR, "Y_tmax_train.npy")
)

Y_tmax_val = np.load(
    os.path.join(PREPARED_DIR, "Y_tmax_val.npy")
)

Y_tmin_train = np.load(
    os.path.join(PREPARED_DIR, "Y_tmin_train.npy")
)

Y_tmin_val = np.load(
    os.path.join(PREPARED_DIR, "Y_tmin_val.npy")
)

spatial_mask = np.load(
    os.path.join(PREPARED_DIR, "spatial_mask.npy")
).astype(np.float32)

mask = tf.constant(
    spatial_mask[..., np.newaxis],
    dtype=tf.float32
)


@tf.keras.utils.register_keras_serializable()
def masked_mse(y_true, y_pred):
    error = tf.square(y_true - y_pred)
    error = error * mask

    return (
        tf.reduce_sum(error, axis=[1, 2, 3])
        / tf.reduce_sum(mask)
    )


@tf.keras.utils.register_keras_serializable()
def masked_mae(y_true, y_pred):
    error = tf.abs(y_true - y_pred)
    error = error * mask

    return (
        tf.reduce_sum(error, axis=[1, 2, 3])
        / tf.reduce_sum(mask)
    )


inputs = layers.Input(
    shape=X_train.shape[1:]
)

x = layers.ConvLSTM2D(
    filters=32,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="tanh"
)(inputs)

x = layers.BatchNormalization()(x)

x = layers.ConvLSTM2D(
    filters=32,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=False,
    activation="tanh"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
)(x)

tmax_output = layers.Conv2D(
    filters=1,
    kernel_size=(1, 1),
    padding="same",
    name="tmax_output"
)(x)

tmin_output = layers.Conv2D(
    filters=1,
    kernel_size=(1, 1),
    padding="same",
    name="tmin_output"
)(x)


model = models.Model(
    inputs=inputs,
    outputs=[
        tmax_output,
        tmin_output
    ]
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss=[
        masked_mse,
        masked_mse
    ],
    metrics=[
        [masked_mae],
        [masked_mae]
    ]
)


early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-5
)


history = model.fit(
    X_train,
    [
        Y_tmax_train[..., np.newaxis],
        Y_tmin_train[..., np.newaxis]
    ],
    validation_data=(
        X_val,
        [
            Y_tmax_val[..., np.newaxis],
            Y_tmin_val[..., np.newaxis]
        ]
    ),
    epochs=50,
    batch_size=4,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=1
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "temperature_convlstm.keras"
)

model.save(MODEL_PATH)


with open(
    os.path.join(
        OUTPUT_DIR,
        "training_history.json"
    ),
    "w"
) as f:

    json.dump(
        {
            key: [float(v) for v in values]
            for key, values in history.history.items()
        },
        f,
        indent=2
    )


print("Temperature ConvLSTM training completed.")
print("Model:", MODEL_PATH)