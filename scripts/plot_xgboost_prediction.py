import os
import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = r"data\processed_daily\cnn_dataset\xgboost"
NC_PATH = r"data\processed_daily\cnn_dataset\common_daily_v2.nc"
MODEL_PATH = r"models\tmax_xgboost.json"
OUTPUT_DIR = r"outputs\xgboost_maps"

os.makedirs(OUTPUT_DIR, exist_ok=True)

import xgboost as xgb

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

pred = model.predict(X_test)

ds = xr.open_dataset(NC_PATH)

lat = ds.latitude.values
lon = ds.longitude.values

target_mask = np.load(
    r"data\processed_daily\cnn_dataset\prepared\target_mask.npy"
).astype(bool)

# Reconstruct prediction maps
n_test = len(X_test)
maps = np.full((n_test, len(lat), len(lon)), np.nan)

valid_indices = np.where(target_mask.ravel())[0]

maps_flat = maps.reshape(n_test, -1)
maps_flat[:, valid_indices] = pred.reshape(n_test, -1)

# Plot a few representative test days
indices = [
    0,
    n_test // 2,
    n_test - 1
]

for i in indices:
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        maps[i],
        origin="lower",
        extent=[
            lon.min(),
            lon.max(),
            lat.min(),
            lat.max()
        ],
        aspect="auto"
    )

    ax.set_title(f"XGBoost Next-Day Tmax Prediction - Test Sample {i}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.colorbar(im, ax=ax, label="Temperature (°C)")

    output_path = os.path.join(
        OUTPUT_DIR,
        f"xgboost_prediction_{i}.png"
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

ds.close()

print("XGBoost prediction maps created.")