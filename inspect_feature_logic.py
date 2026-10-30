
from pathlib import Path
import xarray as xr

path = Path("src/data/prepare_enhanced_convlstm_data.py")
lines = path.read_text(encoding="utf-8").splitlines()

print("=== FEATURE PREPARATION LOGIC ===")
for start, end in [(100, 230), (230, 350)]:
    print(f"\n--- Lines {start}-{end} ---")
    for i in range(start, min(end, len(lines)) + 1):
        print(f"{i}: {lines[i - 1]}")

print("\n=== MASTER DATASET VARIABLES ===")
with xr.open_dataset("data/processed/master_climate_dataset.nc") as ds:
    print("Dimensions:", dict(ds.sizes))
    print("Date range:", str(ds.time.values[0]), "to",
          str(ds.time.values[-1]))
    print("Variables:")
    for name, var in ds.data_vars.items():
        print(f"  {name}: dims={var.dims}, units={var.attrs.get('units', 'unknown')}")
