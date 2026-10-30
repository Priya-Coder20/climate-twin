
import xarray as xr
from pathlib import Path
import numpy as np

files = [
    Path("data/processed_daily/era5/era5_daily.nc"),
    Path("data/live/imd_gridded_daily.nc"),
]

for path in files:
    print("\n" + "=" * 70)
    print("FILE:", path)

    with xr.open_dataset(path) as ds:
        for name, var in ds.data_vars.items():
            values = var.values
            finite = values[np.isfinite(values)]

            print(f"\nVariable: {name}")
            print("  Units:", var.attrs.get("units", "unknown"))
            print("  Shape:", values.shape)

            if finite.size:
                print("  Min:", float(finite.min()))
                print("  Max:", float(finite.max()))
                print("  Mean:", float(finite.mean()))
                print("  Missing values:", int(np.isnan(values).sum()))
            else:
                print("  No finite values found")
