import os
import json
from datetime import datetime, timedelta

import numpy as np


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MOSDAC_DIR = os.path.join(
    BASE_DIR, "data", "live", "mosdac", "processed"
)

ERA5_DIR = os.path.join(
    BASE_DIR, "data", "live", "era5", "processed"
)

IMD_DIR = os.path.join(
    BASE_DIR, "data", "live", "imd_processed"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "live", "model_input"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


FEATURES = [
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


FEATURE_MEAN = np.array([
    5.8697715,
    5.7000237,
    27.038168,
    16.508894,
    1.9696027,
    1005.68774,
    9.522705,
    32.597645,
    19.755241,
    3.231153
], dtype=np.float32)

FEATURE_STD = np.array([
    16.989597,
    8.53692,
    3.3826451,
    5.7244163,
    1.2172827,
    51.484917,
    22.826097,
    0.5255179,
    0.8736007,
    0.9890406
], dtype=np.float32)


MASK_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed_daily",
    "cnn_dataset",
    "multivariable_prepared",
    "spatial_mask.npy"
)

if not os.path.exists(MASK_PATH):
    raise FileNotFoundError(
        f"Spatial mask not found: {MASK_PATH}"
    )

SPATIAL_MASK = np.load(MASK_PATH).astype(bool)


def get_available_dates(directory, prefix):

    if not os.path.exists(directory):
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    dates = set()

    for filename in os.listdir(directory):

        if not filename.startswith(prefix):
            continue

        if not filename.endswith(".npy"):
            continue

        date_text = filename[len(prefix):-4]

        try:
            datetime.strptime(
                date_text,
                "%Y-%m-%d"
            )
            dates.add(date_text)

        except ValueError:
            continue

    return dates


def find_latest_common_dates():

    sources = [
        (MOSDAC_DIR, "imr_"),
        (MOSDAC_DIR, "pet_"),
        (ERA5_DIR, "era5_temperature_"),
        (ERA5_DIR, "era5_dewpoint_"),
        (ERA5_DIR, "era5_wind_speed_"),
        (ERA5_DIR, "era5_pressure_"),
        (ERA5_DIR, "era5_rainfall_"),
        (IMD_DIR, "tmax_"),
        (IMD_DIR, "tmin_"),
        (IMD_DIR, "rainfall_")
    ]

    available_sets = []

    for directory, prefix in sources:

        dates = get_available_dates(
            directory,
            prefix
        )

        if not dates:
            raise RuntimeError(
                f"No files found for {prefix} in {directory}"
            )

        available_sets.append(dates)

    common_dates = set.intersection(
        *available_sets
    )

    if len(common_dates) < 7:
        raise RuntimeError(
            f"Only {len(common_dates)} common dates available. "
            "At least 7 are required."
        )

    sorted_dates = sorted(common_dates)

    latest_window = None

    for i in range(len(sorted_dates) - 6):

        window = sorted_dates[i:i + 7]

        start_date = datetime.strptime(
            window[0],
            "%Y-%m-%d"
        )

        expected_dates = [
            (
                start_date + timedelta(days=j)
            ).strftime("%Y-%m-%d")
            for j in range(7)
        ]

        if window == expected_dates:
            latest_window = window

    if latest_window is None:
        raise RuntimeError(
            "No 7 consecutive common dates found."
        )

    return latest_window


def load_channel(directory, prefix, date_text):

    path = os.path.join(
        directory,
        f"{prefix}{date_text}.npy"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing input file: {path}"
        )

    data = np.load(path).astype(np.float32)

    if data.shape != (65, 73):
        raise ValueError(
            f"Unexpected shape for {path}: "
            f"{data.shape}. Expected (65, 73)."
        )

    return data


def main():

    dates = find_latest_common_dates()

    X = np.zeros(
        (7, 65, 73, 10),
        dtype=np.float32
    )

    for t, date_text in enumerate(dates):

        X[t, :, :, 0] = load_channel(
            MOSDAC_DIR, "imr_", date_text
        )

        X[t, :, :, 1] = load_channel(
            MOSDAC_DIR, "pet_", date_text
        )

        X[t, :, :, 2] = load_channel(
            ERA5_DIR, "era5_temperature_", date_text
        )

        X[t, :, :, 3] = load_channel(
            ERA5_DIR, "era5_dewpoint_", date_text
        )

        X[t, :, :, 4] = load_channel(
            ERA5_DIR, "era5_wind_speed_", date_text
        )

        X[t, :, :, 5] = load_channel(
            ERA5_DIR, "era5_pressure_", date_text
        )

        X[t, :, :, 6] = load_channel(
            ERA5_DIR, "era5_rainfall_", date_text
        )

        X[t, :, :, 7] = load_channel(
            IMD_DIR, "tmax_", date_text
        )

        X[t, :, :, 8] = load_channel(
            IMD_DIR, "tmin_", date_text
        )

        X[t, :, :, 9] = load_channel(
            IMD_DIR, "rainfall_", date_text
        )

    X_normalized = (
        X - FEATURE_MEAN.reshape(1, 1, 1, 10)
    ) / FEATURE_STD.reshape(1, 1, 1, 10)

    invalid_cells = ~SPATIAL_MASK

    for t in range(7):
        for c in range(10):
            X_normalized[t, :, :, c][invalid_cells] = 0.0

    if not np.isfinite(X_normalized).all():
        raise RuntimeError(
            "NaN or infinite values detected."
        )

    prediction_date = (
        datetime.strptime(
            dates[-1],
            "%Y-%m-%d"
        ) + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    output_path = os.path.join(
        OUTPUT_DIR,
        "X_live.npy"
    )

    metadata_path = os.path.join(
        OUTPUT_DIR,
        "X_live_metadata.json"
    )

    np.save(
        output_path,
        X_normalized
    )

    metadata = {
        "input_dates": dates,
        "input_start_date": dates[0],
        "input_end_date": dates[-1],
        "prediction_date": prediction_date,
        "input_shape": list(X_normalized.shape),
        "features": FEATURES,
        "spatial_cells": int(SPATIAL_MASK.sum()),
        "normalization": "frozen_training_statistics"
    }

    with open(metadata_path, "w") as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    print("LIVE MODEL INPUT CREATED")
    print(f"Input dates: {dates[0]} to {dates[-1]}")
    print(f"Prediction date: {prediction_date}")
    print(f"Shape: {X_normalized.shape}")
    print(f"Valid spatial cells: {int(SPATIAL_MASK.sum())}")


if __name__ == "__main__":
    main()