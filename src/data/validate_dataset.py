import xarray as xr
from pathlib import Path

DATASET = Path("data/processed_daily/cnn_dataset/common_daily_v2.nc")

EXPECTED_VARIABLES = [
    "imr",
    "pet",
    "era5_temperature",
    "era5_dewpoint",
    "era5_wind_speed",
    "era5_pressure",
    "era5_rainfall",
    "imd_max_temperature",
    "imd_min_temperature",
    "imd_rainfall",
]


def main():
    print("=" * 70)
    print("MAHARASHTRA CLIMATE DIGITAL TWIN")
    print("MASTER DATASET VALIDATION")
    print("=" * 70)

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    print(f"\nDataset: {DATASET}")
    print(f"Size: {DATASET.stat().st_size / (1024**3):.2f} GB")

    ds = xr.open_dataset(DATASET)

    print("\n--- Dimensions ---")
    for name, size in ds.sizes.items():
        print(f"{name}: {size}")

    print("\n--- Coordinates ---")

    for coord in ["time", "latitude", "longitude"]:
        if coord in ds:
            values = ds[coord].values
            print(f"{coord}:")
            print(f"  first: {values[0]}")
            print(f"  last : {values[-1]}")
            print(f"  count: {len(values)}")

    print("\n--- Variables ---")

    for variable in ds.data_vars:
        data = ds[variable]

        total = data.size
        valid = data.count().item()
        missing = total - valid

        valid_percent = 100 * valid / total
        missing_percent = 100 * missing / total

        print(
            f"{variable:25s} "
            f"valid={valid_percent:7.2f}% "
            f"missing={missing_percent:7.2f}%"
        )

    print("\n--- Expected Variables ---")

    missing_variables = [
        variable
        for variable in EXPECTED_VARIABLES
        if variable not in ds.data_vars
    ]

    extra_variables = [
        variable
        for variable in ds.data_vars
        if variable not in EXPECTED_VARIABLES
    ]

    if missing_variables:
        print("MISSING:")
        for variable in missing_variables:
            print(f"  - {variable}")
    else:
        print("All expected variables are present.")

    if extra_variables:
        print("\nAdditional variables:")
        for variable in extra_variables:
            print(f"  - {variable}")

    print("\n--- Spatial Extent ---")

    print(
        f"Latitude : "
        f"{float(ds.latitude.min()):.6f} "
        f"to "
        f"{float(ds.latitude.max()):.6f}"
    )

    print(
        f"Longitude: "
        f"{float(ds.longitude.min()):.6f} "
        f"to "
        f"{float(ds.longitude.max()):.6f}"
    )

    print("\n--- Dataset Status ---")

    if not missing_variables:
        print("PASS: Expected climate variables are available.")
    else:
        print("FAIL: Expected variables are missing.")

    print("\nValidation complete.")

    ds.close()


if __name__ == "__main__":
    main()