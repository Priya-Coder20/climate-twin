
import json
import re
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "data" / "live"
MASTER = ROOT / "data" / "processed" / "master_climate_dataset.nc"
NORM = ROOT / "data" / "processed" / "enhanced_convlstm_data"
OUT = LIVE / "model_input"

FEATURES = [
    ("imr", LIVE / "mosdac" / "processed", "imr_"),
    ("pet", LIVE / "mosdac" / "processed", "pet_"),
    ("era5_temperature", LIVE / "era5" / "processed", "era5_temperature_"),
    ("era5_dewpoint", LIVE / "era5" / "processed", "era5_dewpoint_"),
    ("era5_wind_speed", LIVE / "era5" / "processed", "era5_wind_speed_"),
    ("era5_pressure", LIVE / "era5" / "processed", "era5_pressure_"),
    ("era5_rainfall", LIVE / "era5" / "processed", "era5_rainfall_"),
    ("imd_max_temperature", LIVE / "imd_processed", "tmax_"),
    ("imd_min_temperature", LIVE / "imd_processed", "tmin_"),
    ("imd_rainfall", LIVE / "imd_processed", "rainfall_"),
]

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.npy$")


def available_dates(folder, prefix):
    dates = set()
    for path in folder.glob(prefix + "*.npy"):
        match = DATE_RE.search(path.name)
        if match:
            dates.add(match.group(1))
    return dates


def load_checked_array(path, name, date, mask):
    arr = np.asarray(np.load(path), dtype=np.float32)

    if arr.shape != mask.shape:
        raise RuntimeError(
            f"{name} on {date}: shape {arr.shape}, expected {mask.shape}"
        )

    inside = arr[mask]

    if not np.isfinite(inside).all():
        raise RuntimeError(
            f"{name} on {date} has missing/non-finite values "
            "inside Maharashtra."
        )

    if np.any(inside <= -900):
        raise RuntimeError(
            f"{name} on {date} contains values <= -900. "
            "Resolve missing-value sentinels before forecasting."
        )

    return arr


def load_day(date, mask):
    arrays = []

    for name, folder, prefix in FEATURES:
        path = folder / f"{prefix}{date}.npy"

        if not path.exists():
            raise RuntimeError(f"Missing {name} for {date}: {path}")

        arrays.append(load_checked_array(path, name, date, mask))

    return np.stack(arrays, axis=-1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(MASTER) as ds:
        mask = ds["maharashtra_mask"].values.astype(bool)
        master_dates = np.datetime_as_string(ds.time.values, unit="D")

        if not mask.any():
            raise RuntimeError("Maharashtra mask contains no usable cells.")

        if "imd_rainfall" not in ds:
            raise RuntimeError("Historical dataset has no imd_rainfall variable.")

        historical_rainfall = ds["imd_rainfall"].values.astype(np.float32)

        # Match historical rainfall dates to their grid cells.
        historical_rain_by_date = {
            str(date): historical_rainfall[i]
            for i, date in enumerate(master_dates)
        }

    date_sets = [
        available_dates(folder, prefix)
        for _, folder, prefix in FEATURES
    ]
    common_dates = set.intersection(*date_sets) if date_sets else set()
    ordered = sorted(common_dates)

    if len(ordered) < 7:
        raise RuntimeError(
            f"Need 7 common dates across all 10 features; found {len(ordered)}."
        )

    chosen = None

    for end in range(len(ordered), 6, -1):
        candidate = ordered[end - 7:end]
        steps = np.diff(np.asarray(candidate, dtype="datetime64[D]"))

        if np.all(steps == np.timedelta64(1, "D")):
            chosen = candidate
            break

    if chosen is None:
        raise RuntimeError(
            "No seven consecutive dates exist across all live features."
        )

    print("Selected input dates:", chosen[0], "to", chosen[-1])

    base = np.stack([load_day(date, mask) for date in chosen], axis=0)

    # Seasonal features, calculated from actual dates.
    day_of_year = np.array(
        [
            (
                np.datetime64(date, "D")
                - np.datetime64(date[:4] + "-01-01", "D")
            ).astype(int) + 1
            for date in chosen
        ],
        dtype=np.float32,
    )

    years = np.array([int(date[:4]) for date in chosen])
    leap = np.array([
        (year % 4 == 0 and year % 100 != 0) or year % 400 == 0
        for year in years
    ])
    year_length = np.where(leap, 366.0, 365.0)
    angle = 2.0 * np.pi * (day_of_year - 1.0) / year_length

    seasonal = np.zeros((7, *mask.shape, 2), dtype=np.float32)
    seasonal[..., 0] = np.sin(angle)[:, None, None]
    seasonal[..., 1] = np.cos(angle)[:, None, None]

    # Use live IMD rainfall first; use historical rainfall only for
    # preceding dates where no live file exists.
    rainfall_cache = {
        date: base[i, ..., 9]
        for i, date in enumerate(chosen)
    }

    def get_rainfall(date):
        if date in rainfall_cache:
            return rainfall_cache[date]

        live_path = LIVE / "imd_processed" / f"rainfall_{date}.npy"

        if live_path.exists():
            arr = load_checked_array(
                live_path, "imd_rainfall", date, mask
            )
        elif date in historical_rain_by_date:
            arr = historical_rain_by_date[date].copy()

            if not np.isfinite(arr[mask]).all():
                raise RuntimeError(
                    f"Historical rainfall for {date} has missing cells."
                )

            if np.any(arr[mask] <= -900):
                raise RuntimeError(
                    f"Historical rainfall for {date} contains a sentinel."
                )
        else:
            raise RuntimeError(
                f"No rainfall file or historical record for {date}; "
                "cannot calculate rainfall accumulations."
            )

        rainfall_cache[date] = arr
        return arr

    rain3 = np.full((7, *mask.shape), np.nan, dtype=np.float32)
    rain7 = np.full((7, *mask.shape), np.nan, dtype=np.float32)

    for i, date in enumerate(chosen):
        current = np.datetime64(date, "D")

        for days, target in ((3, rain3), (7, rain7)):
            dates = [
                str(current - np.timedelta64(offset, "D"))
                for offset in range(days - 1, -1, -1)
            ]
            arrays = [get_rainfall(day) for day in dates]
            stack = np.stack(arrays, axis=0)

            valid = np.isfinite(stack).all(axis=0)
            total = np.sum(stack, axis=0)
            target[i, valid] = total[valid]

    if not np.isfinite(rain3[:, mask]).all():
        raise RuntimeError("3-day rainfall accumulation is incomplete.")

    if not np.isfinite(rain7[:, mask]).all():
        raise RuntimeError("7-day rainfall accumulation is incomplete.")

    features = np.concatenate(
        [base, seasonal, rain3[..., None], rain7[..., None]],
        axis=-1,
    )

    if features.shape != (7, *mask.shape, 14):
        raise RuntimeError(f"Unexpected feature tensor: {features.shape}")

    feature_mean = np.load(NORM / "feature_mean.npy")
    feature_std = np.load(NORM / "feature_std.npy")
    usable = np.load(NORM / "usable_cells.npy").astype(bool)

    if feature_mean.shape != (14,) or feature_std.shape != (14,):
        raise RuntimeError("Expected 14 saved feature normalization values.")

    if not np.isfinite(feature_mean).all():
        raise RuntimeError("Feature means contain non-finite values.")

    if not np.isfinite(feature_std).all() or np.any(feature_std <= 0):
        raise RuntimeError("Feature standard deviations are invalid.")

    if usable.shape != mask.shape or not usable.any():
        raise RuntimeError("Saved usable-cell mask is invalid.")

    if not np.isfinite(features[:, usable, :]).all():
        raise RuntimeError("Input has missing values in usable cells.")

    # Match training convention: zero outside the Maharashtra mask.
    features[:, ~mask, :] = 0.0
    features = (features - feature_mean) / feature_std
    features[:, ~mask, :] = 0.0

    if not np.isfinite(features).all():
        raise RuntimeError("Normalized input contains non-finite values.")

    np.save(OUT / "X_live.npy", features.astype(np.float32))

    metadata = {
        "input_start_date": chosen[0],
        "input_end_date": chosen[-1],
        "shape": list(features.shape),
        "feature_order": [name for name, _, _ in FEATURES] + [
            "sin_season",
            "cos_season",
            "imd_rainfall_3day",
            "imd_rainfall_7day",
        ],
        "normalized": True,
        "is_current": False,
        "note": (
            "Input uses available live feature files and may use historical "
            "rainfall for preceding accumulation days. Validate grid alignment, "
            "units, and source consistency before interpreting predictions."
        ),
    }

    with open(OUT / "X_live_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Input created:", OUT / "X_live.npy")
    print("Shape:", features.shape)
    print("Input dates:", chosen[0], "to", chosen[-1])
    print("Input only; no forecast generated yet.")


if __name__ == "__main__":
    main()
