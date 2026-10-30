import json
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "data" / "processed" / "master_climate_dataset.nc"
OUT = ROOT / "data" / "live" / "grid"
OUT.mkdir(parents=True, exist_ok=True)

URL = "https://api.open-meteo.com/v1/forecast"
TZ = ZoneInfo("Asia/Kolkata")
BATCH_SIZE = 20

HOURLY_VARS = [
    "temperature_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "surface_pressure",
    "precipitation",
    "et0_fao_evapotranspiration",
]

VARIABLES = [
    "tmax_c",
    "tmin_c",
    "temperature_mean_c",
    "dewpoint_mean_c",
    "wind_speed_mean_kmh",
    "surface_pressure_mean_hpa",
    "rainfall_mm",
    "pet_mm",
]


def main():
    if not MASTER.exists():
        raise FileNotFoundError(f"Master grid not found: {MASTER}")

    today = datetime.now(TZ).date()
    print(f"India date: {today}")
    print("Loading existing master grid...")

    with xr.open_dataset(MASTER) as ds:
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        mask_da = ds["maharashtra_mask"]

        if "time" in mask_da.dims:
            mask_da = mask_da.isel(time=0)

        mask = mask_da.transpose("latitude", "longitude").values

    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing="xy")

    if mask.shape != lat_grid.shape:
        raise RuntimeError(
            f"Mask shape {mask.shape} != grid shape {lat_grid.shape}"
        )

    valid = np.isfinite(mask) & (mask > 0)
    iy, ix = np.where(valid)

    print(f"Grid shape: {mask.shape}")
    print(f"Maharashtra cells to fetch: {len(iy)}")

    if len(iy) == 0:
        raise RuntimeError("No valid Maharashtra grid cells found.")

    # Store one grid for each variable and date.
    grids = {}
    failures = []
    session = requests.Session()

    for start in range(0, len(iy), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(iy))
        batch_y = iy[start:end]
        batch_x = ix[start:end]

        lats = lat_grid[batch_y, batch_x]
        lons = lon_grid[batch_y, batch_x]

        params = {
            "latitude": ",".join(f"{v:.5f}" for v in lats),
            "longitude": ",".join(f"{v:.5f}" for v in lons),
            "hourly": ",".join(HOURLY_VARS),
            "past_days": 13,
            "forecast_days": 7,
            "timezone": "Asia/Kolkata",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
        }

        print(
            f"Fetching grid batch {start // BATCH_SIZE + 1}/"
            f"{(len(iy) + BATCH_SIZE - 1) // BATCH_SIZE} "
            f"({len(lats)} cells)"
        )

        
        payload = None

        for attempt in range(6):
            response = None

            try:
                response = session.get(
                    URL, params=params, timeout=120
                )

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")

                    try:
                        wait_seconds = (
                            max(1, min(int(retry_after), 600))
                            if retry_after
                            else min(15 * (2 ** attempt), 300)
                        )
                    except ValueError:
                        wait_seconds = min(15 * (2 ** attempt), 300)

                    print(f"  Rate limited. Waiting {wait_seconds}s...")
                    time.sleep(wait_seconds)
                    continue

                response.raise_for_status()
                payload = response.json()
                break

            except (requests.RequestException, ValueError) as exc:
                if attempt == 5:
                    failures.append({
                        "batch_start": start,
                        "batch_end": end,
                        "error": str(exc),
                    })
                    print(f"  Batch failed: {exc}")
                else:
                    wait_seconds = min(15 * (2 ** attempt), 300)
                    print(f"  Temporary error. Waiting {wait_seconds}s...")
                    time.sleep(wait_seconds)

        if payload is None:
            if not any(
                f.get("batch_start") == start
                and f.get("batch_end") == end
                for f in failures
            ):
                failures.append({
                    "batch_start": start,
                    "batch_end": end,
                    "error": "No response after retries, possibly rate limited",
                })
            continue


        # Multi-coordinate requests normally return a list of responses.
        results = payload if isinstance(payload, list) else [payload]

        if len(results) != len(lats):
            failures.append({
                "batch_start": start,
                "batch_end": end,
                "error": (
                    f"Expected {len(lats)} locations; "
                    f"received {len(results)}"
                ),
            })
            print("  Batch response count mismatch; skipping batch.")
            continue

        for j, result in enumerate(results):
            hourly = result.get("hourly", {})
            times = hourly.get("time", [])

            if not times:
                continue

            frame = pd.DataFrame({"time": times})
            frame["time"] = pd.to_datetime(frame["time"])
            frame["date"] = frame["time"].dt.strftime("%Y-%m-%d")

            source_names = {
                "tmax_c": ("temperature_2m", "max"),
                "tmin_c": ("temperature_2m", "min"),
                "temperature_mean_c": ("temperature_2m", "mean"),
                "dewpoint_mean_c": ("dew_point_2m", "mean"),
                "wind_speed_mean_kmh": ("wind_speed_10m", "mean"),
                "surface_pressure_mean_hpa": (
                    "surface_pressure", "mean"
                ),
                "rainfall_mm": ("precipitation", "sum"),
                "pet_mm": ("et0_fao_evapotranspiration", "sum"),
            }

            for output_name, (source_name, operation) in source_names.items():
                values = hourly.get(source_name)

                if values is None or len(values) != len(times):
                    frame[output_name] = np.nan
                else:
                    frame[output_name] = pd.to_numeric(
                        pd.Series(values), errors="coerce"
                    ).to_numpy()

            # Convert hourly records into daily grid-cell values.
            aggregations = {
                name: operation
                for name, (_, operation) in source_names.items()
            }

            daily = frame.groupby("date").agg(aggregations)

            for day, row in daily.iterrows():
                key = (day,)

                if key not in grids:
                    grids[key] = {
                        name: np.full(mask.shape, np.nan, dtype=np.float32)
                        for name in VARIABLES
                    }

                for name in VARIABLES:
                    value = row[name]
                    if pd.notna(value):
                        grids[key][name][batch_y[j], batch_x[j]] = (
                            float(value)
                        )

        time.sleep(3)

    if not grids:
        raise RuntimeError(
            "No grid data was retrieved. Check API errors and retry."
        )

    # Save each daily variable as a 65 x 73 NumPy array.
    saved_dates = []
    for (day,), variable_grids in sorted(grids.items()):
        day_dir = OUT / day
        day_dir.mkdir(parents=True, exist_ok=True)

        for name, array in variable_grids.items():
            # Outside-Maharashtra cells remain NaN.
            np.save(day_dir / f"{name}.npy", array)

        saved_dates.append(day)

    metadata = {
        "source": "Open-Meteo Forecast API",
        "retrieved_at": datetime.now(TZ).isoformat(),
        "india_today": today.isoformat(),
        "grid_shape": list(mask.shape),
        "maharashtra_cells": int(valid.sum()),
        "variables": VARIABLES,
        "saved_dates": saved_dates,
        "failed_batches": failures,
        "important_note": (
            "These are Open-Meteo model-based estimates/forecasts, "
            "not verified IMD observations. PET is reference "
            "evapotranspiration (ET0), not necessarily the PET "
            "variable used during training. This is not yet the "
            "complete 14-feature ConvLSTM input."
        ),
    }

    (OUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Grid shape: {mask.shape}")
    print(f"Dates saved: {len(saved_dates)}")
    if saved_dates:
        print(f"First date: {saved_dates[0]}")
        print(f"Last date: {saved_dates[-1]}")
    print(f"Failed batches: {len(failures)}")
    print(f"Output: {OUT}")

    if failures:
        print("Some batches failed. Check metadata.json before proceeding.")


if __name__ == "__main__":
    main()