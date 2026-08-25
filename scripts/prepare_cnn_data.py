import os
import numpy as np

INPUT_DIR = r"data\processed_daily\cnn_dataset\sequences"
OUTPUT_DIR = r"data\processed_daily\cnn_dataset\prepared"

os.makedirs(OUTPUT_DIR, exist_ok=True)

X = np.load(
    os.path.join(INPUT_DIR, "X_temperature.npy")
)

Y = np.load(
    os.path.join(INPUT_DIR, "Y_temperature.npy")
)

n = len(X)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X[:train_end]
X_val = X[train_end:val_end]
X_test = X[val_end:]

Y_train = Y[:train_end]
Y_val = Y[train_end:val_end]
Y_test = Y[val_end:]

mean = X_train.mean(axis=(0, 1, 2, 3))
std = X_train.std(axis=(0, 1, 2, 3))

std[std == 0] = 1

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)

np.save(os.path.join(OUTPUT_DIR, "Y_train.npy"), Y_train)
np.save(os.path.join(OUTPUT_DIR, "Y_val.npy"), Y_val)
np.save(os.path.join(OUTPUT_DIR, "Y_test.npy"), Y_test)

np.save(os.path.join(OUTPUT_DIR, "feature_mean.npy"), mean)
np.save(os.path.join(OUTPUT_DIR, "feature_std.npy"), std)

print("CNN data prepared.")
print("Train:", X_train.shape, Y_train.shape)
print("Validation:", X_val.shape, Y_val.shape)
print("Test:", X_test.shape, Y_test.shape)