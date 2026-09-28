import os
import numpy as np
import xarray as xr
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

XGB_DIR = r"data\processed_daily\cnn_dataset\xgboost"
NC_PATH = r"data\processed_daily\cnn_dataset\common_daily_v2.nc"
MODEL_PATH = r"models\tmax_xgboost.json"
MASK_PATH = r"data\processed_daily\cnn_dataset\prepared\target_mask.npy"

OUTPUT_DIR = r"outputs\xgboost_error_maps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load test data
X_test = np.load(
    os.path.join(XGB_DIR, "X_test.npy")
)

Y_test = np.load(
    os.path.join(XGB_DIR, "Y_test.npy")
)

# Load model
model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

# Predictions
pred = model.predict(X_test)

# -------------------------------------------------
# Reconstruct (test_days, valid_spatial_cells)
# -------------------------------------------------

n_test_days = 207
valid_cells = np.load(MASK_PATH).sum()

errors = np.abs(
    Y_test - pred
)

errors = errors.reshape(
    n_test_days,
    valid_cells
)

# MAE for each valid spatial cell
spatial_mae_valid = np.mean(
    errors,
    axis=0
)

# -------------------------------------------------
# Put spatial MAE back onto 65 × 73 grid
# -------------------------------------------------

mask = np.load(MASK_PATH).astype(bool)

spatial_mae = np.full(
    mask.shape,
    np.nan,
    dtype=np.float32
)

spatial_mae[mask] = spatial_mae_valid

# -------------------------------------------------
# Coordinates
# -------------------------------------------------

ds = xr.open_dataset(NC_PATH)

lat = ds.latitude.values
lon = ds.longitude.values

# -------------------------------------------------
# Plot
# -------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 7))

im = ax.imshow(
    spatial_mae,
    origin="lower",
    extent=[
        lon.min(),
        lon.max(),
        lat.min(),
        lat.max()
    ],
    aspect="auto"
)

ax.set_title(
    "XGBoost Spatial MAE - Next-Day Tmax"
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

fig.colorbar(
    im,
    ax=ax,
    label="MAE (°C)"
)

output_path = os.path.join(
    OUTPUT_DIR,
    "xgboost_spatial_mae.png"
)

fig.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)
ds.close()

print(f"Spatial MAE map saved: {output_path}")
print(f"Mean spatial MAE: {np.nanmean(spatial_mae):.4f} °C")
print(f"Minimum spatial MAE: {np.nanmin(spatial_mae):.4f} °C")
print(f"Maximum spatial MAE: {np.nanmax(spatial_mae):.4f} °C")