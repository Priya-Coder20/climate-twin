import os
import numpy as np

BASE_DIR = r"C:\Users\HP\climate-twin"

MASK_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed_daily",
    "cnn_dataset",
    "multivariable_prepared",
    "spatial_mask.npy"
)

MASK = np.load(MASK_FILE).astype(bool)

DATES = [
    "2026-09-14",
    "2026-09-15",
    "2026-09-16",
    "2026-09-17",
    "2026-09-18",
    "2026-09-19",
    "2026-09-20"
]

CHANNELS = {
    "imr": os.path.join(BASE_DIR, "data", "live", "mosdac", "processed"),
    "pet": os.path.join(BASE_DIR, "data", "live", "mosdac", "processed"),
    "era5_temperature": os.path.join(BASE_DIR, "data", "live", "era5", "processed"),
    "era5_dewpoint": os.path.join(BASE_DIR, "data", "live", "era5", "processed"),
    "era5_wind_speed": os.path.join(BASE_DIR, "data", "live", "era5", "processed"),
    "era5_pressure": os.path.join(BASE_DIR, "data", "live", "era5", "processed"),
    "era5_rainfall": os.path.join(BASE_DIR, "data", "live", "era5", "processed"),
    "imd_max_temperature": os.path.join(BASE_DIR, "data", "live", "imd_processed"),
    "imd_min_temperature": os.path.join(BASE_DIR, "data", "live", "imd_processed"),
    "imd_rainfall": os.path.join(BASE_DIR, "data", "live", "imd_processed")
}

PREFIX = {
    "imr": "imr_",
    "pet": "pet_",
    "era5_temperature": "era5_temperature_",
    "era5_dewpoint": "era5_dewpoint_",
    "era5_wind_speed": "era5_wind_speed_",
    "era5_pressure": "era5_pressure_",
    "era5_rainfall": "era5_rainfall_",
    "imd_max_temperature": "tmax_",
    "imd_min_temperature": "tmin_",
    "imd_rainfall": "rainfall_"
}

for date in DATES:

    print("\nDate:", date)

    for channel in CHANNELS:

        path = os.path.join(
            CHANNELS[channel],
            PREFIX[channel] + date + ".npy"
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {channel}: {path}"
            )

        data = np.load(path).astype(np.float32)

        if data.shape != (65, 73):
            raise ValueError(
                f"{channel}: wrong shape {data.shape}"
            )

        values = data[MASK]

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{channel}: NaN/Inf inside model mask"
            )

        print(
            f"{channel:22s}",
            f"shape={data.shape}",
            f"min={values.min():.3f}",
            f"max={values.max():.3f}",
            f"mean={values.mean():.3f}"
        )

print("\nALL 10 LIVE CHANNELS VALID.")
print("Grid: 65 x 73")
print("Valid spatial cells:", MASK.sum())
print("Time steps:", len(DATES))