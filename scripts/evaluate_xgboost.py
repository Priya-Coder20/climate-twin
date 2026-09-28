import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = r"data\processed_daily\cnn_dataset\xgboost"
MODEL_PATH = r"models\tmax_xgboost.json"

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
Y_test = np.load(os.path.join(DATA_DIR, "Y_test.npy"))

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

pred = model.predict(X_test)

mae = mean_absolute_error(Y_test, pred)
rmse = np.sqrt(mean_squared_error(Y_test, pred))
r2 = r2_score(Y_test, pred)

print(f"XGBoost MAE:  {mae:.4f} °C")
print(f"XGBoost RMSE: {rmse:.4f} °C")
print(f"XGBoost R²:   {r2:.4f}")