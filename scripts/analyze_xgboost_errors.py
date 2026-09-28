import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

XGB_DIR = r"data\processed_daily\cnn_dataset\xgboost"
SEQUENCE_DIR = r"data\processed_daily\cnn_dataset"
MODEL_PATH = r"models\tmax_xgboost.json"

# XGBoost test data
X_test_xgb = np.load(
    os.path.join(XGB_DIR, "X_test.npy")
)
Y_test_xgb = np.load(
    os.path.join(XGB_DIR, "Y_test.npy")
)

# Original unstandardized sequence data
X_all = np.load(
    os.path.join(SEQUENCE_DIR, "X_temperature.npy")
)
Y_all = np.load(
    os.path.join(SEQUENCE_DIR, "Y_temperature.npy")
)

target_mask = np.load(
    os.path.join(SEQUENCE_DIR, "spatial_mask.npy")
).astype(bool)

# -------------------------------------------------
# XGBoost prediction
# -------------------------------------------------

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

xgb_pred = model.predict(X_test_xgb)

xgb_mae = mean_absolute_error(
    Y_test_xgb,
    xgb_pred
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        Y_test_xgb,
        xgb_pred
    )
)

# -------------------------------------------------
# Persistence baseline
# -------------------------------------------------

# Test set = final 207 samples
n_test = len(Y_test_xgb)

X_test_original = X_all[-n_test:]
Y_test_original = Y_all[-n_test:]

# Last historical day
# Variable 7 = IMD maximum temperature
persistence_pred = X_test_original[:, -1, :, :, 7]

# Valid spatial cells
actual = Y_test_original[:, target_mask]
persistence = persistence_pred[:, target_mask]

actual = actual.reshape(-1)
persistence = persistence.reshape(-1)

persistence_mae = mean_absolute_error(
    actual,
    persistence
)

persistence_rmse = np.sqrt(
    mean_squared_error(
        actual,
        persistence
    )
)

# -------------------------------------------------
# Comparison
# -------------------------------------------------

mae_improvement = persistence_mae - xgb_mae
rmse_improvement = persistence_rmse - xgb_rmse

mae_percent = (
    mae_improvement / persistence_mae
) * 100

rmse_percent = (
    rmse_improvement / persistence_rmse
) * 100

print(f"XGBoost MAE:          {xgb_mae:.4f} °C")
print(f"Persistence MAE:      {persistence_mae:.4f} °C")
print(f"MAE improvement:      {mae_improvement:.4f} °C")
print(f"MAE improvement (%):  {mae_percent:.2f}%")

print()

print(f"XGBoost RMSE:         {xgb_rmse:.4f} °C")
print(f"Persistence RMSE:     {persistence_rmse:.4f} °C")
print(f"RMSE improvement:     {rmse_improvement:.4f} °C")
print(f"RMSE improvement (%):  {rmse_percent:.2f}%")