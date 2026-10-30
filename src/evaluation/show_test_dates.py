import numpy as np
import xarray as xr

DATA_PATH = r"data\processed\master_climate_dataset.nc"

INPUT_DAYS = 7
FORECAST_DAYS = 7

ds = xr.open_dataset(DATA_PATH)
dates = ds.time.values

valid_forecast_windows = []

for i in range(
    len(dates) - INPUT_DAYS - FORECAST_DAYS + 1
):
    window_dates = dates[i:i + INPUT_DAYS + FORECAST_DAYS]

    if not np.all(
        np.diff(window_dates) == np.timedelta64(1, "D")
    ):
        continue

    valid_forecast_windows.append(
        window_dates[INPUT_DAYS:
                     INPUT_DAYS + FORECAST_DAYS]
    )

n = len(valid_forecast_windows)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

test_windows = valid_forecast_windows[val_end:]

print("Test windows:", len(test_windows))
print()
print("Test forecast date range:")
print(test_windows[0][0], "to", test_windows[-1][-1])
print()
print("Available forecast dates:")

for window in test_windows:
    print(
        window[0],
        "to",
        window[-1]
    )

ds.close()