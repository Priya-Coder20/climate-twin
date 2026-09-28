import os
import numpy as np
import tensorflow as tf
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr


# ============================================================
# PATHS
# ============================================================

PREPARED_DIR = r"data\processed_daily\cnn_dataset\prepared"
XGB_DIR = r"data\processed_daily\cnn_dataset\xgboost"
MODEL_DIR = r"models"
OUTPUT_DIR = r"outputs\spatial_error_comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD TEST DATA
# ============================================================

X_test = np.load(os.path.join(PREPARED_DIR, "X_test.npy"))
Y_test = np.load(os.path.join(PREPARED_DIR, "Y_test.npy"))
target_mask = np.load(os.path.join(PREPARED_DIR, "target_mask.npy"))

print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)
print("Valid cells:", np.sum(target_mask))


# ============================================================
# GRID INFORMATION
# ============================================================

ds = xr.open_dataset(
    r"data\processed_daily\cnn_dataset\common_daily_v2.nc"
)

lat = ds["latitude"].values
lon = ds["longitude"].values

n_test = X_test.shape[0]
n_lat = len(lat)
n_lon = len(lon)


# ============================================================
# 1. CNN
# ============================================================

print("\nRunning CNN predictions...")

cnn_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "tmax_cnn.keras"),
    compile=False
)

# CNN expects:
# (samples, latitude, longitude, 70)
X_test_cnn = X_test.reshape(
    n_test,
    n_lat,
    n_lon,
    -1
)

cnn_pred = cnn_model.predict(
    X_test_cnn,
    batch_size=8,
    verbose=1
)

cnn_pred = np.squeeze(cnn_pred)

cnn_errors = np.abs(
    cnn_pred[:, target_mask] -
    Y_test[:, target_mask]
)

cnn_spatial_mae = np.mean(cnn_errors, axis=0)

cnn_mean_mae = np.mean(cnn_errors)

print("CNN mean spatial MAE:", round(float(cnn_mean_mae), 4), "°C")


# ============================================================
# 2. ConvLSTM
# ============================================================

print("\nRunning ConvLSTM predictions...")

convlstm_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "tmax_convlstm.keras"),
    compile=False
)

convlstm_pred = convlstm_model.predict(
    X_test,
    batch_size=4,
    verbose=1
)

convlstm_pred = np.squeeze(convlstm_pred)

convlstm_errors = np.abs(
    convlstm_pred[:, target_mask] -
    Y_test[:, target_mask]
)

convlstm_spatial_mae = np.mean(
    convlstm_errors,
    axis=0
)

convlstm_mean_mae = np.mean(convlstm_errors)

print(
    "ConvLSTM mean spatial MAE:",
    round(float(convlstm_mean_mae), 4),
    "°C"
)


# ============================================================
# 3. XGBOOST
# ============================================================

print("\nRunning XGBoost predictions...")

X_test_xgb = np.load(
    os.path.join(XGB_DIR, "X_test.npy")
)

xgb_model = xgb.XGBRegressor()

xgb_model.load_model(
    os.path.join(MODEL_DIR, "tmax_xgboost.json")
)

xgb_pred = xgb_model.predict(X_test_xgb)

# Each test day has 2380 valid cells
xgb_pred = xgb_pred.reshape(
    n_test,
    np.sum(target_mask)
)

y_test_xgb = Y_test[:, target_mask]

xgb_errors = np.abs(
    xgb_pred - y_test_xgb
)

xgb_spatial_mae = np.mean(
    xgb_errors,
    axis=0
)

xgb_mean_mae = np.mean(xgb_errors)

print(
    "XGBoost mean spatial MAE:",
    round(float(xgb_mean_mae), 4),
    "°C"
)


# ============================================================
# CONVERT SPATIAL MAE TO 65 x 73 GRID
# ============================================================

cnn_map = np.full(
    (n_lat, n_lon),
    np.nan
)

convlstm_map = np.full(
    (n_lat, n_lon),
    np.nan
)

xgb_map = np.full(
    (n_lat, n_lon),
    np.nan
)

cnn_map[target_mask] = cnn_spatial_mae
convlstm_map[target_mask] = convlstm_spatial_mae
xgb_map[target_mask] = xgb_spatial_mae


# ============================================================
# COMMON COLOR SCALE
# ============================================================

all_values = np.concatenate([
    cnn_spatial_mae,
    convlstm_spatial_mae,
    xgb_spatial_mae
])

vmin = np.min(all_values)
vmax = np.max(all_values)


# ============================================================
# PLOT FUNCTION
# ============================================================

def save_map(data, title, filename):

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    im = ax.imshow(
        data,
        origin="lower",
        extent=[
            lon.min(),
            lon.max(),
            lat.min(),
            lat.max()
        ],
        aspect="auto",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Spatial MAE (°C)")

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# SAVE INDIVIDUAL MAPS
# ============================================================

save_map(
    cnn_map,
    "CNN Spatial MAE",
    "cnn_spatial_mae.png"
)

save_map(
    convlstm_map,
    "ConvLSTM Spatial MAE",
    "convlstm_spatial_mae.png"
)

save_map(
    xgb_map,
    "XGBoost Spatial MAE",
    "xgboost_spatial_mae.png"
)


# ============================================================
# SAVE NUMERICAL COMPARISON
# ============================================================

comparison_file = os.path.join(
    OUTPUT_DIR,
    "spatial_mae_comparison.csv"
)

with open(comparison_file, "w") as f:

    f.write(
        "Model,Mean_Spatial_MAE_C,Minimum_Spatial_MAE_C,Maximum_Spatial_MAE_C\n"
    )

    f.write(
        f"CNN,{np.mean(cnn_spatial_mae):.4f},"
        f"{np.min(cnn_spatial_mae):.4f},"
        f"{np.max(cnn_spatial_mae):.4f}\n"
    )

    f.write(
        f"ConvLSTM,{np.mean(convlstm_spatial_mae):.4f},"
        f"{np.min(convlstm_spatial_mae):.4f},"
        f"{np.max(convlstm_spatial_mae):.4f}\n"
    )

    f.write(
        f"XGBoost,{np.mean(xgb_spatial_mae):.4f},"
        f"{np.min(xgb_spatial_mae):.4f},"
        f"{np.max(xgb_spatial_mae):.4f}\n"
    )


# ============================================================
# FINAL
# ============================================================

print("\nSpatial comparison completed.")

print("\nResults:")
print(
    f"CNN      : {np.mean(cnn_spatial_mae):.4f} °C"
)
print(
    f"ConvLSTM : {np.mean(convlstm_spatial_mae):.4f} °C"
)
print(
    f"XGBoost  : {np.mean(xgb_spatial_mae):.4f} °C"
)

print(
    "\nOutputs saved to:",
    OUTPUT_DIR
)