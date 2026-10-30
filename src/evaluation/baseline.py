import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = r"data\processed\model_data"
OUTPUT_DIR = r"outputs\metrics"

os.makedirs(OUTPUT_DIR, exist_ok=True)

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
Y_test = np.load(os.path.join(DATA_DIR, "Y_test.npy"))

# Last observed day from the input sequence
# X shape: samples × input_days × cells × features
last_day = X_test[:, -1, :, :]

# Target variables are:
# 7 = IMD max temperature
# 8 = IMD min temperature
# 9 = IMD rainfall
target_feature_indices = [7, 8, 9]

prediction = last_day[:, :, target_feature_indices]

# Repeat the latest observation for all 7 forecast days
prediction = np.repeat(
    prediction[:, np.newaxis, :, :],
    Y_test.shape[1],
    axis=1
)

results = []

target_names = [
    "tmax",
    "tmin",
    "rainfall"
]

for i, name in enumerate(target_names):

    actual = Y_test[:, :, :, i].reshape(-1)
    predicted = prediction[:, :, :, i].reshape(-1)

    mask = np.isfinite(actual) & np.isfinite(predicted)

    actual = actual[mask]
    predicted = predicted[mask]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    results.append(
        [name, mae, rmse, r2]
    )

np.savetxt(
    os.path.join(OUTPUT_DIR, "baseline_metrics.csv"),
    np.array(results, dtype=object),
    delimiter=",",
    fmt="%s",
    header="variable,MAE,RMSE,R2",
    comments=""
)

print("Baseline evaluation complete.")
for row in results:
    print(
        f"{row[0]}: "
        f"MAE={row[1]:.4f}, "
        f"RMSE={row[2]:.4f}, "
        f"R2={row[3]:.4f}"
    )