
import json
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "data" / "processed" / "master_climate_dataset.nc"
GRID_DIR = ROOT / "data" / "live" / "grid"
OUTPUT_DIR = ROOT / "data" / "live" / "dashboard"
OUTPUT_FILE = OUTPUT_DIR / "live_weather.json"

VARIABLES = {
    "tmax_c": "Maximum temperature",
    "tmin_c": "Minimum temperature",
    "rainfall_mm": "Rainfall",
}


def json_grid(array, mask):
    """Convert a grid to JSON-compatible values; outside Maharashtra = null."""
    result = np.asarray(array, dtype=np.float32).copy()
    result[~mask] = np.nan
    result[~np.isfinite(result)] = np.nan
    return [
        [None if not np.isfinite(v) else round(float(v), 3) for v in row]
        for row in result
    ]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = GRID_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {metadata_path}. Run fetch_live_grid.py first."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    today = metadata.get("india_today")
    if not today:
        raise RuntimeError("Grid metadata does not contain india_today.")

    with xr.open_dataset(MASTER) as ds:
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        mask_da = ds["maharashtra_mask"]

        if "time" in mask_da.dims:
            mask_da = mask_da.isel(time=0)

        mask = (
            mask_da.transpose("latitude", "longitude").values > 0
        )

    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing="xy")

    if mask.shape != (len(lat), len(lon)):
        raise RuntimeError("Master grid coordinate and mask shapes disagree.")

    # Include today's estimate and future forecast dates only.
    dates = sorted(
        d for d in metadata.get("saved_dates", [])
        if d >= today
    )

    if not dates:
        raise RuntimeError(
            "No current/future dates found. Refresh the live grid first."
        )

    daily_data = []

    for day in dates:
        day_dir = GRID_DIR / day
        layers = {}

        for variable, label in VARIABLES.items():
            path = day_dir / f"{variable}.npy"

            if not path.exists():
                raise FileNotFoundError(f"Missing grid: {path}")

            arr = np.load(path).astype(np.float32)

            if arr.shape != mask.shape:
                raise RuntimeError(
                    f"{path.name} shape {arr.shape}; expected {mask.shape}"
                )

            if not np.isfinite(arr[mask]).all():
                raise RuntimeError(
                    f"{path.name} has missing values inside Maharashtra."
                )

            values = arr[mask]
            if variable in ("tmax_c", "tmin_c"):
                unit = "°C"
            else:
                unit = "mm"

            layers[variable] = {
                "label": label,
                "unit": unit,
                "grid": json_grid(arr, mask),
                "min": round(float(values.min()), 2),
                "max": round(float(values.max()), 2),
                "mean": round(float(values.mean()), 2),
            }

        daily_data.append({
            "date": day,
            "data_type": (
                "current_day_estimate" if day == today
                else "forecast"
            ),
            "layers": layers,
        })

    payload = {
        "title": "Maharashtra Live Weather",
        "source": "Open-Meteo Forecast API",
        "retrieved_at": metadata.get("retrieved_at"),
        "india_today": today,
        "grid": {
            "rows": int(len(lat)),
            "columns": int(len(lon)),
            "latitude": lat.tolist(),
            "longitude": lon.tolist(),
            "latitude_grid": lat_grid.tolist(),
            "longitude_grid": lon_grid.tolist(),
            "maharashtra_mask": mask.tolist(),
        },
        "available_layers": list(VARIABLES.keys()),
        "dates": daily_data,
        "disclaimer": (
            "Open-Meteo model-based estimates and forecasts; "
            "not verified IMD station observations or ConvLSTM predictions."
        ),
    }

    OUTPUT_FILE.write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )

    print("Dashboard data created successfully.")
    print("Output:", OUTPUT_FILE)
    print("Today:", today)
    print("Dates included:", ", ".join(dates))
    print("Grid:", len(lat), "x", len(lon))
    print("Maharashtra cells:", int(mask.sum()))
    print("Layers:", ", ".join(VARIABLES.keys()))
    print("Source: Open-Meteo")
    print("This file contains provider weather data, not AI predictions.")


if __name__ == "__main__":
    main()
