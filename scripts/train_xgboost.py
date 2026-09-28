import os
import numpy as np
import xgboost as xgb

DATA_DIR = r"data\processed_daily\cnn_dataset\xgboost"
MODEL_DIR = r"models"

os.makedirs(MODEL_DIR, exist_ok=True)

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))
X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
Y_val = np.load(os.path.join(DATA_DIR, "Y_val.npy"))

model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

model.fit(
    X_train,
    Y_train,
    eval_set=[(X_val, Y_val)],
    verbose=False
)

model.save_model(
    os.path.join(MODEL_DIR, "tmax_xgboost.json")
)

print("XGBoost training completed.")