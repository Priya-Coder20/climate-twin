from pathlib import Path
import xarray as xr

files = list(Path("data").rglob("*.nc"))

print(f"NetCDF files found: {len(files)}")

for path in files:
    try:
        with xr.open_dataset(path) as ds:
            time_keys = [
                key for key in ds.coords
                if key.lower() in ("time", "valid_time", "date")
            ]

            if time_keys:
                times = ds[time_keys[0]].values
                if len(times):
                    print(
                        f"{path} | "
                        f"{str(times[0])[:10]} to "
                        f"{str(times[-1])[:10]} | "
                        f"{len(times)} time steps"
                    )
                else:
                    print(f"{path} | Empty time coordinate")
            else:
                print(f"{path} | No recognized time coordinate")

    except Exception as exc:
        print(f"{path} | Could not inspect: {exc}")