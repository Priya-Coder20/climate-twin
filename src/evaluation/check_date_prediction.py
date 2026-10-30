import os
import numpy as np
import xarray as xr
import tensorflow as tf

DATA_PATH = r"data\processed\master_climate_dataset.nc"
DATA_DIR = r"data\processed\convlstm_data"
MODEL_PATH = r"models\trained\convlstm_forecaster.keras"

INPUT_DAYS = 7
FORECAST_DAYS = 7

date = input("Enter forecast date (YYYY-MM-DD): ").strip()

ds = xr.open_dataset(DATA_PATH)

dates = ds.time.values
mask = np.load(os.path.join(DATA_DIR, "maharashtra_mask.npy"))

features = np.stack(
    [
        ds["imr"].values,
        ds["pet"].values,
        ds["era5_temperature"].values,
        ds["era5_dewpoint"].values,
        ds["era5_wind_speed"].values,
        ds["era5_pressure"].values,
        ds["era5_rainfall"].values,
        ds["imd_max_temperature"].values,
        ds["imd_min_temperature"].values,
        ds["imd_rainfall"].values
    ],
    axis=-1
)

targets = np.stack(
    [
        ds["imd_max_temperature"].values,
        ds["imd_min_temperature"].values,
        ds["imd_rainfall"].values
    ],
    axis=-1
)

usable_cells = np.load(
    os.path.join(DATA_DIR, "usable_cells.npy")
)

X = []
Y = []
forecast_dates = []

for i in range(
    len(dates) - INPUT_DAYS - FORECAST_DAYS + 1
):

    window_dates = dates[
        i:i + INPUT_DAYS + FORECAST_DAYS
    ]

    if not np.all(
        np.diff(window_dates) == np.timedelta64(1, "D")
    ):
        continue

    x = features[i:i + INPUT_DAYS].copy()

    y = targets[
        i + INPUT_DAYS:
        i + INPUT_DAYS + FORECAST_DAYS
    ].copy()

    if not np.isfinite(
        x[:, usable_cells, :]
    ).all():
        continue

    if not np.isfinite(
        y[:, usable_cells, :]
    ).all():
        continue

    x[:, ~mask, :] = 0
    y[:, ~mask, :] = 0

    X.append(x)
    Y.append(y)

    forecast_dates.append(
        window_dates[INPUT_DAYS:
        INPUT_DAYS + FORECAST_DAYS]
    )

X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.float32)

forecast_dates = np.asarray(forecast_dates)

n = len(X)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_test = X[val_end:]
Y_test = Y[val_end:]
test_dates = forecast_dates[val_end:]

matches = []

for i, window in enumerate(test_dates):
    if date in window.astype("datetime64[D]").astype(str):
        matches.append((i, window))

if not matches:
    print("Date is not available as a forecast date in the test set.")
    ds.close()
    raise SystemExit

sample_index, window = matches[0]

forecast_day = list(
    window.astype("datetime64[D]").astype(str)
).index(date)

X_sample = X_test[sample_index:sample_index + 1]

target_mean = np.load(
    os.path.join(DATA_DIR, "target_mean.npy")
)

target_std = np.load(
    os.path.join(DATA_DIR, "target_std.npy")
)

feature_mean = np.load(
    os.path.join(DATA_DIR, "feature_mean.npy")
)

feature_std = np.load(
    os.path.join(DATA_DIR, "feature_std.npy")
)

X_sample = (
    X_sample - feature_mean
) / feature_std

X_sample = np.nan_to_num(
    X_sample,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

model = tf.keras.models.load_model(MODEL_PATH)

prediction = model.predict(
    X_sample,
    verbose=0
)

prediction = (
    prediction * target_std
    + target_mean
)

actual = (
    Y_test[sample_index]
    * target_std
    + target_mean
)

pred_tmax = np.nanmean(
    prediction[0, forecast_day, mask]
)

actual_tmax = np.nanmean(
    actual[forecast_day, mask]
)

pred_tmin = np.nanmean(
    prediction[0, forecast_day, mask]
)

actual_tmin = np.nanmean(
    actual[forecast_day, mask]
)

print()
print(f"Date: {date}")
print(f"Forecast day: {forecast_day + 1}")
print()
print(f"Tmax Actual:    {actual_tmax:.2f} °C")
print(f"Tmax Predicted: {pred_tmax:.2f} °C")
print(f"Tmax Error:     {abs(actual_tmax - pred_tmax):.2f} °C")
print()
print(f"Tmin Actual:    {actual_tmin:.2f} °C")
print(f"Tmin Predicted: {pred_tmin:.2f} °C")
print(f"Tmin Error:     {abs(actual_tmin - pred_tmin):.2f} °C")

ds.close()