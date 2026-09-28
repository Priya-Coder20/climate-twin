import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ============================================================
# PATHS
# ============================================================

DATA_DIR = r"data\processed_daily\cnn_dataset\clean_prepared"
MODEL_DIR = r"models"
OUTPUT_DIR = r"outputs\convlstm_clean"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "tmax_convlstm_clean.keras"
)

# ============================================================
# SETTINGS
# ============================================================

EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 0.001

ROWS = 65
COLS = 73

# ============================================================
# LOAD DATA
# ============================================================

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))

X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
Y_val = np.load(os.path.join(DATA_DIR, "Y_val.npy"))

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
Y_test = np.load(os.path.join(DATA_DIR, "Y_test.npy"))

MASK = np.load(
    os.path.join(DATA_DIR, "target_mask.npy")
)

FEATURES = X_train.shape[-1]


# ============================================================
# RESTORE SPATIAL GRID
# ============================================================

def restore_grid(X):
    samples, time_steps, valid_cells, features = X.shape

    full = np.zeros(
        (samples, time_steps, ROWS, COLS, features),
        dtype=np.float32
    )

    for i in range(samples):
        for t in range(time_steps):
            full[i, t, MASK, :] = X[i, t]

    return full


def restore_target(Y):
    samples = Y.shape[0]

    full = np.zeros(
        (samples, ROWS, COLS),
        dtype=np.float32
    )

    full[:, MASK] = Y

    return full


X_train = restore_grid(X_train)
X_val = restore_grid(X_val)
X_test = restore_grid(X_test)

Y_train = restore_target(Y_train)
Y_val = restore_target(Y_val)
Y_test = restore_target(Y_test)


# ============================================================
# MASKED LOSS
# ============================================================

mask_float = MASK.astype(np.float32)


def masked_mse(y_true, y_pred):

    error = tf.square(y_true - y_pred)

    mask_tensor = tf.constant(
        mask_float,
        dtype=tf.float32
    )

    error = error * mask_tensor

    return tf.reduce_sum(error) / (
        tf.reduce_sum(mask_tensor)
        + tf.keras.backend.epsilon()
    )


def masked_mae(y_true, y_pred):

    error = tf.abs(y_true - y_pred)

    mask_tensor = tf.constant(
        mask_float,
        dtype=tf.float32
    )

    error = error * mask_tensor

    return tf.reduce_sum(error) / (
        tf.reduce_sum(mask_tensor)
        + tf.keras.backend.epsilon()
    )


# ============================================================
# CONVLSTM MODEL
# ============================================================

model = models.Sequential([

    layers.Input(
        shape=(7, ROWS, COLS, FEATURES)
    ),

    layers.ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="tanh"
    ),

    layers.BatchNormalization(),

    layers.ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=False,
        activation="tanh"
    ),

    layers.BatchNormalization(),

    layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    ),

    layers.Conv2D(
        filters=1,
        kernel_size=(1, 1),
        padding="same",
        activation="linear"
    ),

    layers.Lambda(
	lambda x: tf.squeeze(x,axis=-1)
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss=masked_mse,
    metrics=[masked_mae]
)


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

checkpoint = callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ],
    verbose=1
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(MODEL_PATH)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

np.savez(
    os.path.join(
        OUTPUT_DIR,
        "training_history.npz"
    ),
    loss=np.array(history.history["loss"]),
    val_loss=np.array(history.history["val_loss"]),
    mae=np.array(history.history["masked_mae"]),
    val_mae=np.array(
        history.history["val_masked_mae"]
    )
)


# ============================================================
# EVALUATION
# ============================================================

val_results = model.evaluate(
    X_val,
    Y_val,
    batch_size=BATCH_SIZE,
    verbose=0
)

test_results = model.evaluate(
    X_test,
    Y_test,
    batch_size=BATCH_SIZE,
    verbose=0
)

print("\nCLEAN CONVLSTM RESULTS")
print("======================")
print(f"Validation MAE: {val_results[1]:.4f} °C")
print(f"Test MAE:       {test_results[1]:.4f} °C")
print("======================")
print(f"Model saved: {MODEL_PATH}")