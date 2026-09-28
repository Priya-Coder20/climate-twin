import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

INPUT_DIR = r"data\processed_daily\cnn_dataset\multivariable_prepared"
MODEL_DIR = r"models"
OUTPUT_DIR = r"outputs\multivariable_convlstm"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.001
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

X_train = np.load(
    os.path.join(INPUT_DIR, "X_train.npy")
)

X_val = np.load(
    os.path.join(INPUT_DIR, "X_val.npy")
)

Y_tmax_train = np.load(
    os.path.join(INPUT_DIR, "Y_tmax_train.npy")
)

Y_tmax_val = np.load(
    os.path.join(INPUT_DIR, "Y_tmax_val.npy")
)

Y_tmin_train = np.load(
    os.path.join(INPUT_DIR, "Y_tmin_train.npy")
)

Y_tmin_val = np.load(
    os.path.join(INPUT_DIR, "Y_tmin_val.npy")
)

Y_rain_train = np.load(
    os.path.join(INPUT_DIR, "Y_rainfall_train.npy")
)

Y_rain_val = np.load(
    os.path.join(INPUT_DIR, "Y_rainfall_val.npy")
)

spatial_mask = np.load(
    os.path.join(INPUT_DIR, "spatial_mask.npy")
).astype(np.float32)

if spatial_mask.shape != (65, 73):
    raise ValueError(
        f"Unexpected spatial mask shape: {spatial_mask.shape}"
    )

if int(spatial_mask.sum()) != 2379:
    raise ValueError(
        f"Unexpected number of valid spatial cells: "
        f"{int(spatial_mask.sum())}"
    )

SPATIAL_MASK = tf.constant(
    spatial_mask,
    dtype=tf.float32
)


@tf.keras.utils.register_keras_serializable()
def masked_mse(y_true, y_pred):
    squared_error = tf.square(
        y_true - y_pred
    )

    masked_error = (
        squared_error * SPATIAL_MASK
    )

    return (
        tf.reduce_sum(
            masked_error,
            axis=[1, 2]
        )
        /
        tf.reduce_sum(SPATIAL_MASK)
    )


@tf.keras.utils.register_keras_serializable()
def masked_mae(y_true, y_pred):
    absolute_error = tf.abs(
        y_true - y_pred
    )

    masked_error = (
        absolute_error * SPATIAL_MASK
    )

    return (
        tf.reduce_sum(
            masked_error,
            axis=[1, 2]
        )
        /
        tf.reduce_sum(SPATIAL_MASK)
    )


Y_train = {
    "tmax_output": Y_tmax_train,
    "tmin_output": Y_tmin_train,
    "rainfall_output": Y_rain_train
}

Y_val = {
    "tmax_output": Y_tmax_val,
    "tmin_output": Y_tmin_val,
    "rainfall_output": Y_rain_val
}

inputs = layers.Input(
    shape=(7, 65, 73, 10),
    name="climate_input"
)

x = layers.ConvLSTM2D(
    32,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="tanh",
    recurrent_activation="sigmoid"
)(inputs)

x = layers.BatchNormalization()(x)

x = layers.ConvLSTM2D(
    32,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=False,
    activation="tanh",
    recurrent_activation="sigmoid"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Conv2D(
    32,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.Conv2D(
    16,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
)(x)

tmax_output = layers.Conv2D(
    1,
    kernel_size=(1, 1),
    activation="linear",
    name="tmax"
)(x)

tmin_output = layers.Conv2D(
    1,
    kernel_size=(1, 1),
    activation="linear",
    name="tmin"
)(x)

rain_output = layers.Conv2D(
    1,
    kernel_size=(1, 1),
    activation="linear",
    name="rainfall"
)(x)

tmax_output = layers.Reshape(
    (65, 73),
    name="tmax_output"
)(tmax_output)

tmin_output = layers.Reshape(
    (65, 73),
    name="tmin_output"
)(tmin_output)

rain_output = layers.Reshape(
    (65, 73),
    name="rainfall_output"
)(rain_output)

model = models.Model(
    inputs=inputs,
    outputs=[
        tmax_output,
        tmin_output,
        rain_output
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss={
        "tmax_output": masked_mse,
        "tmin_output": masked_mse,
        "rainfall_output": masked_mse
    },
    loss_weights={
        "tmax_output": 1.0,
        "tmin_output": 1.0,
        "rainfall_output": 1.0
    },
    metrics={
        "tmax_output": [masked_mae],
        "tmin_output": [masked_mae],
        "rainfall_output": [masked_mae]
    }
)

model_path = os.path.join(
    MODEL_DIR,
    "multivariable_convlstm.keras"
)

checkpoint = callbacks.ModelCheckpoint(
    model_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    X_train,
    Y_train,
    validation_data=(
        X_val,
        Y_val
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],
    shuffle=False
)

history_path = os.path.join(
    OUTPUT_DIR,
    "training_history.json"
)

with open(history_path, "w") as f:
    json.dump(
        {
            key: [
                float(v)
                for v in values
            ]
            for key, values
            in history.history.items()
        },
        f,
        indent=2
    )

model.save(model_path)

print("Masked multivariable ConvLSTM training completed.")