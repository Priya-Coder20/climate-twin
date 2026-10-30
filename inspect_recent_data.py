
from pathlib import Path
import xarray as xr

files = [
    Path("data/processed_daily/era5/era5_daily.nc"),
    Path("data/live/imd_gridded_daily.nc"),
    Path("data/live/era5/2026-09-20/era5.nc"),
]

for path in files:
    print("\n" + "=" * 65)
    print("FILE:", path)

    if not path.exists():
        print("File does not exist.")
        continue

    try:
        with xr.open_dataset(path) as ds:
            print("\nDimensions:", dict(ds.sizes))
            print("\nCoordinates:")
            for name, coord in ds.coords.items():
                values = coord.values
                if values.size:
                    print(
                        f"  {name}: shape={values.shape}, "
                        f"first={str(values.flat[0])[:40]}, "
                        f"last={str(values.flat[-1])[:40]}"
                    )

            print("\nData variables:")
            for name, var in ds.data_vars.items():
                print(
                    f"  {name}: dims={var.dims}, "
                    f"shape={var.shape}, units={var.attrs.get('units', 'unknown')}"
                )

    except Exception as exc:
        print("Could not open with xarray:", repr(exc))
