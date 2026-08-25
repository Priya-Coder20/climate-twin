import os
import numpy as np
import xarray as xr

INPUT = r"data\processed_daily\cnn_dataset\common_daily.nc"
OUTPUT_DIR = r"data\processed_daily\cnn_dataset\sequences"

os.makedirs(OUTPUT_DIR, exist_ok=True)

ds = xr.open_dataset(INPUT)

features = [
    "imr",
    "pet",
    "era5_temperature",
    "era5_dewpoint",
    "era5_wind_speed",
    "era5_pressure",
    "era5_rainfall",
    "imd_max_temperature",
    "imd_min_temperature",
    "imd_rainfall"
]

data = ds[features].to_array().transpose(
    "time", "latitude", "longitude", "variable"
).values

target = ds["imd_max_temperature"].values

data = np.nan_to_num(data, nan=0.0)
target = np.nan_to_num(target, nan=0.0)

window = 7

X = []
Y = []

for i in range(len(data) - window):
    X.append(data[i:i + window])
    Y.append(target[i + window])

X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.float32)

np.save(
    os.path.join(OUTPUT_DIR, "X_temperature.npy"),
    X
)

np.save(
    os.path.join(OUTPUT_DIR, "Y_temperature.npy"),
    Y
)

ds.close()

print("CNN sequences created.")
print("X shape:", X.shape)
print("Y shape:", Y.shape)