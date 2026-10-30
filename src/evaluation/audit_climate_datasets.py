
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]

FILES = [
    ROOT / "data" / "processed" / "master_climate_dataset_v2.nc",
    ROOT / "data" / "processed_daily" / "cnn_dataset" / "common_daily_v2.nc",
    ROOT / "data" / "processed_daily" / "cnn_dataset" / "common_daily.nc",
    ROOT / "data" / "processed_daily" / "era5" / "era5_daily.nc",
    ROOT / "data" / "processed_daily" / "imd" / "imd_rainfall_daily.nc",
    ROOT / "data" / "processed_daily" / "imd" / "imd_temperature_daily.nc",
]


def inspect_dataset(path):
    print("\n" + "=" * 90)
    print(f"FILE: {path.relative_to(ROOT)}")
    print("=" * 90)

    if not path.exists():
        print("STATUS: FILE NOT FOUND")
        return

    try:
        with xr.open_dataset(path) as ds:
            print("\nDimensions:")
            print(dict(ds.sizes))

            print("\nCoordinates:")
            for name, coord in ds.coords.items():
                print(
                    f"  {name}: dims={coord.dims}, "
                    f"shape={coord.shape}, "
                    f"units={coord.attrs.get('units', 'not specified')}"
                )

                if coord.size and coord.size <= 20:
                    print(f"    values={coord.values}")

                if name.lower() in ("time", "date", "valid_time"):
                    try:
                        values = coord.values
                        print(
                            f"    first={values.min()}, "
                            f"last={values.max()}"
                        )
                    except (TypeError, ValueError):
                        pass

            print("\nData variables:")
            for name, da in ds.data_vars.items():
                print(f"\n  Variable: {name}")
                print(f"    Dimensions: {da.dims}")
                print(f"    Shape: {da.shape}")
                print(
                    f"    Units: {da.attrs.get('units', 'not specified')}"
                )
                print(
                    f"    Long name: "
                    f"{da.attrs.get('long_name', 'not specified')}"
                )
                print(
                    f"    Fill value metadata: "
                    f"{da.attrs.get('_FillValue', 'not specified')}"
                )

                if not np.issubdtype(da.dtype, np.number):
                    print(f"    Non-numeric dtype: {da.dtype}")
                    continue

                # Read one variable at a time to limit memory use.
                values = np.asarray(da.values)
                finite = np.isfinite(values)

                print(f"    Dtype: {values.dtype}")
                print(f"    Total values: {values.size:,}")
                print(f"    NaN/Inf values: {np.count_nonzero(~finite):,}")

                if values.size == 0 or not finite.any():
                    print("    No finite values to summarize.")
                    continue

                valid = values[finite]
                print(f"    Min: {valid.min():.6g}")
                print(f"    Max: {valid.max():.6g}")
                print(f"    Mean: {valid.mean():.6g}")
                print(f"    Exact -999 count: {np.count_nonzero(valid == -999):,}")
                print(f"    Exact -9999 count: {np.count_nonzero(valid == -9999):,}")

                for attr in ("scale_factor", "add_offset", "missing_value"):
                    if attr in da.attrs:
                        print(f"    {attr}: {da.attrs[attr]}")

    except Exception as exc:
        print(f"ERROR reading dataset: {type(exc).__name__}: {exc}")


def main():
    for path in FILES:
        inspect_dataset(path)

    print("\n" + "=" * 90)
    print("AUDIT FINISHED — no files were modified.")
    print("=" * 90)


if __name__ == "__main__":
    main()
