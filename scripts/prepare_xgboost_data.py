import os
import numpy as np

DATA_DIR = r"data\processed_daily\cnn_dataset\prepared"
OUTPUT_DIR = r"data\processed_daily\cnn_dataset\xgboost"

os.makedirs(OUTPUT_DIR, exist_ok=True)

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))
X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
Y_val = np.load(os.path.join(DATA_DIR, "Y_val.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
Y_test = np.load(os.path.join(DATA_DIR, "Y_test.npy"))
target_mask = np.load(
    os.path.join(DATA_DIR, "target_mask.npy")
).astype(bool)


def prepare_features(X, Y):
    # X: (samples, 7, 65, 73, 10)
    # Y: (samples, 65, 73)

    n_samples = X.shape[0]

    # Select only valid spatial cells first
    X = X[:, :, target_mask, :]

    # X is now: (samples, 7, valid_cells, 10)
    # Rearrange to: (samples, valid_cells, 7, 10)
    X = np.transpose(X, (0, 2, 1, 3))

    # Each spatial cell becomes one training row
    # 7 days × 10 variables = 70 features
    X = X.reshape(-1, 7 * 10)

    # Corresponding target temperature
    Y = Y[:, target_mask].reshape(-1)

    valid = np.isfinite(X).all(axis=1) & np.isfinite(Y)

    return X[valid].astype(np.float32), Y[valid].astype(np.float32)


X_train_xgb, Y_train_xgb = prepare_features(X_train, Y_train)
X_val_xgb, Y_val_xgb = prepare_features(X_val, Y_val)
X_test_xgb, Y_test_xgb = prepare_features(X_test, Y_test)

np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train_xgb)
np.save(os.path.join(OUTPUT_DIR, "Y_train.npy"), Y_train_xgb)
np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val_xgb)
np.save(os.path.join(OUTPUT_DIR, "Y_val.npy"), Y_val_xgb)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test_xgb)
np.save(os.path.join(OUTPUT_DIR, "Y_test.npy"), Y_test_xgb)

print("XGBoost data prepared.")
print("Train:", X_train_xgb.shape)
print("Validation:", X_val_xgb.shape)
print("Test:", X_test_xgb.shape)