
import json
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "data" / "processed" / "master_climate_dataset.nc"
OUT = ROOT / "data" / "live" / "grid"

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

with xr.open_dataset(MASTER) as ds:
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    mask = ds["maharashtra_mask"]

    if "time" in mask.dims:
        mask = mask.isel(time=0)

    mask = mask.transpose("latitude", "longitude").values

valid = np.isfinite(mask) & (mask > 0)
lon_grid, lat_grid = np.meshgrid(lon, lat, indexing="xy")
lats = lat_grid[valid]
lons = lon_grid[valid]


print("Grid shape:", mask.shape)
print("Maharashtra cells:", int(valid.sum()))
print(f"Latitude: {lats.min():.3f} to {lats.max():.3f}")
print(f"Longitude: {lons.min():.3f} to {lons.max():.3f}")

if not (
    lats.min() > 14 and lats.max() < 24
    and lons.min() > 71 and lons.max() < 82
):
    raise SystemExit("FAIL: coordinates are outside expected Maharashtra bounds.")

metadata_path = OUT / "metadata.json"
if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    print("Source:", metadata.get("source"))
    print("Dates recorded:", len(metadata.get("saved_dates", [])))
    print("Failed batches:", len(metadata.get("failed_batches", [])))
    print("Coordinate bounds recorded:",
          metadata.get("coordinate_bounds", "Not recorded"))
else:
    print("WARNING: metadata.json not found.")

day_dirs = sorted(
    p for p in OUT.iterdir()
    if p.is_dir() and len(p.name) == 10
)

print("Date folders:", len(day_dirs))
all_ok = True

for day_dir in day_dirs:
    missing_files = []
    for name in VARIABLES:
        path = day_dir / f"{name}.npy"
        if not path.exists():
            missing_files.append(name)

    if missing_files:
        all_ok = False
        print(day_dir.name, "MISSING FILES:", missing_files)
        continue

    arrays = {
        name: np.load(day_dir / f"{name}.npy")
        for name in VARIABLES
    }

    print(f"\n{day_dir.name}")
    for name, arr in arrays.items():
        if arr.shape != mask.shape:
            all_ok = False
            print(f"  FAIL {name}: shape {arr.shape}")
            continue

        vals = arr[valid]
        finite = np.isfinite(vals)
        coverage = finite.mean() * 100 if len(vals) else 0

        if finite.any():
            v = vals[finite]
            print(
                f"  {name}: coverage={coverage:.1f}% "
                f"min={v.min():.2f} max={v.max():.2f}"
            )
        else:
            print(f"  {name}: NO VALID VALUES")
            all_ok = False

        if coverage < 99:
            all_ok = False

        # Outside-Maharashtra cells should remain NaN.
        if np.isfinite(arr[~valid]).any():
            all_ok = False
            print(f"    WARNING: values outside Maharashtra mask")


if all_ok and len(day_dirs) > 0:
    print("Structural and coverage checks passed.")
    print("Still verify API coordinate matching and variable compatibility.")
else:
    print("CHECK FAILED: inspect missing files, shapes, or coverage above.")
