
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path("data/live")
DATE = "2026-09-20"

FILES = {
    "imr": ROOT / "mosdac" / "processed" / f"imr_{DATE}.npy",
    "pet": ROOT / "mosdac" / "processed" / f"pet_{DATE}.npy",
    "era5_temperature": ROOT / "era5" / "processed" / f"era5_temperature_{DATE}.npy",
    "era5_dewpoint": ROOT / "era5" / "processed" / f"era5_dewpoint_{DATE}.npy",
    "era5_wind_speed": ROOT / "era5" / "processed" / f"era5_wind_speed_{DATE}.npy",
    "era5_pressure": ROOT / "era5" / "processed" / f"era5_pressure_{DATE}.npy",
    "era5_rainfall": ROOT / "era5" / "processed" / f"era5_rainfall_{DATE}.npy",
    "imd_max_temperature": ROOT / "imd_processed" / f"tmax_{DATE}.npy",
    "imd_min_temperature": ROOT / "imd_processed" / f"tmin_{DATE}.npy",
    "imd_rainfall": ROOT / "imd_processed" / f"rainfall_{DATE}.npy",
}

print("=" * 72)
print("LIVE FEATURE VALIDATION")
print("Date:", DATE)
print("=" * 72)

live_arrays = {}

for name, path in FILES.items():
    print(f"\n{name}:")
    print("  File:", path)

    if not path.exists():
        print("  MISSING")
        continue

    arr = np.asarray(np.load(path), dtype=np.float64)
    live_arrays[name] = arr
    finite = np.isfinite(arr)

    print("  Shape:", arr.shape)
    print("  Finite cells:", f"{finite.sum()}/{arr.size}")
    print("  Finite coverage:", f"{100 * finite.mean():.2f}%")

    if finite.any():
        values = arr[finite]
        print("  Min:", float(values.min()))
        print("  Max:", float(values.max()))
        print("  Mean:", float(values.mean()))

print("\n" + "=" * 72)
print("TRAINING GRID AND REFERENCE RANGES")
print("=" * 72)

master_path = Path("data/processed/master_climate_dataset.nc")

with xr.open_dataset(master_path) as ds:
    print("Training grid:", (ds.sizes["latitude"], ds.sizes["longitude"]))
    print("Training dates:", str(ds.time.values[0]), "to",
          str(ds.time.values[-1]))

    for name in FILES:
        if name not in ds.data_vars or name not in live_arrays:
            continue

        live = live_arrays[name]
        historical = ds[name].values

        print(f"\n{name}:")
        print("  Live shape:", live.shape)
        print("  Historical grid shape:", historical.shape[-2:])

        if live.shape != historical.shape[-2:]:
            print("  WARNING: GRID SHAPE MISMATCH")
            continue

        historical_values = historical[np.isfinite(historical)]
        if historical_values.size:
            print("  Historical overall min:", float(historical_values.min()))
            print("  Historical overall max:", float(historical_values.max()))
            print("  Historical overall mean:", float(historical_values.mean()))

        recent = historical[-30:]
        recent_values = recent[np.isfinite(recent)]
        if recent_values.size:
            print("  Historical last-30-day min:", float(recent_values.min()))
            print("  Historical last-30-day max:", float(recent_values.max()))
            print("  Historical last-30-day mean:", float(recent_values.mean()))

print("\nValidation report complete. No files were modified.")
