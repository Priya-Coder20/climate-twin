import os
import json
import numpy as np
import xarray as xr

DATA_PATH = r"data\processed_daily\cnn_dataset\common_daily_v2.nc"
OUTPUT_DIR = r"data\processed_daily\cnn_dataset\multivariable_sequences"

os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_VARIABLES = [
    "imr",
    "pet",
    "era5_temperature",
    "era5_dewpoint",
    "era5_wind_speed",
    "era5_pressure",
    "era5_rainfall",
    "imd_max_temperature",
    "imd_min_temperature",
    "imd_rainfall",
]

TARGET_VARIABLES = [
    "imd_max_temperature",
    "imd_min_temperature",
    "imd_rainfall",
]

SEQUENCE_LENGTH = 7


ds = xr.open_dataset(DATA_PATH)

times = ds.time.values
n_times = len(times)

lat = ds.latitude.values
lon = ds.longitude.values

data = {
    var: ds[var].values.astype(np.float32)
    for var in INPUT_VARIABLES
}

ds.close()

height = len(lat)
width = len(lon)


def valid_values(arr):
    return np.isfinite(arr) & (arr != -999)


imd_mask = np.isfinite(
    data["imd_max_temperature"][0]
)


valid_masks = {
    var: valid_values(data[var])
    for var in INPUT_VARIABLES
}


stable_masks = []

for var in INPUT_VARIABLES:

    if var == "pet":
        continue

    always_valid = np.all(
        valid_masks[var],
        axis=0
    )

    stable_masks.append(always_valid)


stable_mask = (
    imd_mask
    & np.logical_and.reduce(stable_masks)
)


pet_invalid = ~valid_masks["pet"]

persistent_pet_invalid = np.all(
    pet_invalid,
    axis=0
)

persistent_pet_invalid &= imd_mask

stable_mask &= ~persistent_pet_invalid


pet = data["pet"].copy()

pet_missing_before = int(
    np.sum(
        (~valid_values(pet))[:, stable_mask]
    )
)


for y, x in zip(
    *np.where(stable_mask)
):

    series = pet[:, y, x]

    valid = (
        np.isfinite(series)
        & (series != -999)
    )

    if not np.any(valid):
        continue

    if np.any(~valid):

        valid_indices = np.where(valid)[0]
        missing_indices = np.where(~valid)[0]

        series[missing_indices] = np.interp(
            missing_indices,
            valid_indices,
            series[valid_indices]
        )

        pet[:, y, x] = series


data["pet"] = pet


pet_missing_after = int(
    np.sum(
        (~valid_values(data["pet"]))[
            :, stable_mask
        ]
    )
)


possible_windows = (
    n_times - SEQUENCE_LENGTH
)

X_sequences = []
Y_tmax_sequences = []
Y_tmin_sequences = []
Y_rainfall_sequences = []
target_dates = []

discarded_temporal_gap_windows = 0
discarded_missing_input_windows = 0
discarded_missing_target_windows = 0


for start in range(possible_windows):

    end = start + SEQUENCE_LENGTH
    target_index = end

    window_times = times[
        start:target_index + 1
    ]

    date_numbers = (
        window_times
        .astype("datetime64[D]")
        .astype(np.int64)
    )

    if not np.all(
        np.diff(date_numbers) == 1
    ):
        discarded_temporal_gap_windows += 1
        continue


    input_valid = True

    for var in INPUT_VARIABLES:

        window = data[var][
            start:end
        ]

        invalid = ~valid_values(window)

        if np.any(
            invalid[:, stable_mask]
        ):
            input_valid = False
            break


    if not input_valid:
        discarded_missing_input_windows += 1
        continue


    targets_valid = True

    for var in TARGET_VARIABLES:

        target = data[var][target_index]

        invalid = ~valid_values(target)

        if np.any(
            invalid[stable_mask]
        ):
            targets_valid = False
            break


    if not targets_valid:
        discarded_missing_target_windows += 1
        continue


    X = np.empty(
        (
            SEQUENCE_LENGTH,
            height,
            width,
            len(INPUT_VARIABLES)
        ),
        dtype=np.float32
    )


    for feature_index, var in enumerate(
        INPUT_VARIABLES
    ):

        X[:, :, :, feature_index] = data[var][
            start:end
        ]


    Y_tmax = data[
        "imd_max_temperature"
    ][target_index].copy()

    Y_tmin = data[
        "imd_min_temperature"
    ][target_index].copy()

    Y_rainfall = data[
        "imd_rainfall"
    ][target_index].copy()


    X[:, ~stable_mask, :] = np.nan

    Y_tmax[~stable_mask] = np.nan
    Y_tmin[~stable_mask] = np.nan
    Y_rainfall[~stable_mask] = np.nan


    X_sequences.append(X)
    Y_tmax_sequences.append(Y_tmax)
    Y_tmin_sequences.append(Y_tmin)
    Y_rainfall_sequences.append(Y_rainfall)

    target_dates.append(
        str(times[target_index])[:10]
    )


X_sequences = np.asarray(
    X_sequences,
    dtype=np.float32
)

Y_tmax_sequences = np.asarray(
    Y_tmax_sequences,
    dtype=np.float32
)

Y_tmin_sequences = np.asarray(
    Y_tmin_sequences,
    dtype=np.float32
)

Y_rainfall_sequences = np.asarray(
    Y_rainfall_sequences,
    dtype=np.float32
)

target_dates = np.asarray(
    target_dates
)


if len(X_sequences) > 0:

    x_invalid = int(
        np.sum(
            ~np.isfinite(
                X_sequences[
                    :, :, stable_mask, :
                ]
            )
        )
    )

    tmax_invalid = int(
        np.sum(
            ~np.isfinite(
                Y_tmax_sequences[
                    :, stable_mask
                ]
            )
        )
    )

    tmin_invalid = int(
        np.sum(
            ~np.isfinite(
                Y_tmin_sequences[
                    :, stable_mask
                ]
            )
        )
    )

    rainfall_invalid = int(
        np.sum(
            ~np.isfinite(
                Y_rainfall_sequences[
                    :, stable_mask
                ]
            )
        )
    )

else:

    x_invalid = 0
    tmax_invalid = 0
    tmin_invalid = 0
    rainfall_invalid = 0


if len(target_dates) > 1:

    target_numbers = (
        target_dates
        .astype("datetime64[D]")
        .astype(np.int64)
    )

    largest_target_gap = int(
        np.max(
            np.diff(target_numbers)
        )
    )

else:

    largest_target_gap = None


np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_multivariable.npy"
    ),
    X_sequences
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "Y_tmax.npy"
    ),
    Y_tmax_sequences
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "Y_tmin.npy"
    ),
    Y_tmin_sequences
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "Y_rainfall.npy"
    ),
    Y_rainfall_sequences
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "spatial_mask.npy"
    ),
    stable_mask
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "target_dates.npy"
    ),
    target_dates
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "feature_names.npy"
    ),
    np.asarray(INPUT_VARIABLES)
)


metadata = {
    "source_dataset": DATA_PATH,
    "sequence_length_days": SEQUENCE_LENGTH,
    "input_variables": INPUT_VARIABLES,
    "target_variables": TARGET_VARIABLES,
    "grid_height": int(height),
    "grid_width": int(width),
    "total_dates": int(n_times),
    "imd_valid_cells": int(imd_mask.sum()),
    "persistent_pet_invalid_cells": int(
        persistent_pet_invalid.sum()
    ),
    "final_spatial_cells": int(
        stable_mask.sum()
    ),
    "possible_windows": int(
        possible_windows
    ),
    "valid_sequences": int(
        len(X_sequences)
    ),
    "discarded_temporal_gap_windows": int(
        discarded_temporal_gap_windows
    ),
    "discarded_missing_input_windows": int(
        discarded_missing_input_windows
    ),
    "discarded_missing_target_windows": int(
        discarded_missing_target_windows
    ),
    "pet_missing_before_interpolation": int(
        pet_missing_before
    ),
    "pet_missing_after_interpolation": int(
        pet_missing_after
    ),
    "x_invalid_inside_mask": x_invalid,
    "tmax_invalid_inside_mask": tmax_invalid,
    "tmin_invalid_inside_mask": tmin_invalid,
    "rainfall_invalid_inside_mask": rainfall_invalid,
    "largest_target_date_gap": largest_target_gap,
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "metadata.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


print("Multivariable sequence generation completed.")
print()
print("Spatial mask:", int(stable_mask.sum()), "cells")
print("Possible windows:", possible_windows)
print("Valid sequences:", len(X_sequences))
print("Temporal-gap windows:", discarded_temporal_gap_windows)
print("Missing-input windows:", discarded_missing_input_windows)
print("Missing-target windows:", discarded_missing_target_windows)
print()
print("X shape:", X_sequences.shape)
print("Tmax shape:", Y_tmax_sequences.shape)
print("Tmin shape:", Y_tmin_sequences.shape)
print("Rainfall shape:", Y_rainfall_sequences.shape)
print()
print("Invalid X values:", x_invalid)
print("Invalid Tmax values:", tmax_invalid)
print("Invalid Tmin values:", tmin_invalid)
print("Invalid rainfall values:", rainfall_invalid)
print("PET missing before interpolation:", pet_missing_before)
print("PET missing after interpolation:", pet_missing_after)