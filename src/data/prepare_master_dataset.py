from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import Point


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATASET = Path(
    "data/processed_daily/cnn_dataset/common_daily_v2.nc"
)

BOUNDARY = Path(
    "data/boundary/maharashtra_wgs84.shp"
)

OUTPUT = Path(
    "data/processed/master_climate_dataset.nc"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

VARIABLES = [
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


# ---------------------------------------------------------
# Build spatial mask
# ---------------------------------------------------------

def build_maharashtra_mask(ds, boundary):

    latitudes = ds["latitude"].values
    longitudes = ds["longitude"].values

    polygon = boundary.geometry.union_all()

    mask = np.zeros(
        (len(latitudes), len(longitudes)),
        dtype=bool,
    )

    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            point = Point(float(lon), float(lat))

            # Covers includes points on the state boundary.
            mask[i, j] = polygon.covers(point)

    return xr.DataArray(
        mask,
        coords={
            "latitude": ds.latitude,
            "longitude": ds.longitude,
        },
        dims=("latitude", "longitude"),
        name="maharashtra_mask",
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("MAHARASHTRA CLIMATE DIGITAL TWIN")
    print("MASTER DATASET PREPARATION")
    print("=" * 70)

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET}"
        )

    if not BOUNDARY.exists():
        raise FileNotFoundError(
            f"Boundary not found: {BOUNDARY}"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLoading climate dataset...")
    ds = xr.open_dataset(DATASET)

    print("Loading Maharashtra boundary...")
    boundary = gpd.read_file(BOUNDARY)

    print(f"Boundary CRS: {boundary.crs}")

    if boundary.crs is None:
        raise ValueError(
            "Maharashtra boundary has no CRS."
        )

    if boundary.crs.to_epsg() != 4326:
        print("Reprojecting boundary to EPSG:4326...")
        boundary = boundary.to_crs("EPSG:4326")

    # -----------------------------------------------------
    # Validate variables
    # -----------------------------------------------------

    missing_variables = [
        variable
        for variable in VARIABLES
        if variable not in ds.data_vars
    ]

    if missing_variables:
        raise ValueError(
            "Missing variables: "
            + ", ".join(missing_variables)
        )

    print("\nAll required variables are present.")

    # -----------------------------------------------------
    # Build state mask
    # -----------------------------------------------------

    print("\nBuilding Maharashtra spatial mask...")

    mask = build_maharashtra_mask(
        ds,
        boundary,
    )

    total_cells = mask.size
    state_cells = int(mask.sum().item())

    print(
        f"Grid cells inside Maharashtra: "
        f"{state_cells:,} / {total_cells:,}"
    )

    print(
        f"Spatial coverage: "
        f"{100 * state_cells / total_cells:.2f}%"
    )

    # -----------------------------------------------------
    # Apply state mask
    # -----------------------------------------------------

    print("\nApplying Maharashtra mask...")

    clean = ds[VARIABLES].where(mask)

    # -----------------------------------------------------
    # Calculate common valid cells
    # -----------------------------------------------------

    print("\nChecking common spatial coverage...")

    common_valid = xr.ones_like(
        clean[VARIABLES[0]].isel(time=0),
        dtype=bool,
    )

    for variable in VARIABLES:

        valid_anywhere = clean[variable].notnull().any(
            dim="time"
        )

        common_valid = common_valid & valid_anywhere

    common_cells = int(common_valid.sum().item())

    print(
        f"Cells with at least one valid observation "
        f"for every variable: {common_cells:,}"
    )

    # -----------------------------------------------------
    # Dataset attributes
    # -----------------------------------------------------

    clean.attrs = {
        "title": (
            "Multi-source Climate Dataset "
            "for Maharashtra Digital Twin"
        ),
        "region": "Maharashtra, India",
        "crs": "EPSG:4326",
        "source_dataset": str(DATASET),
        "state_mask": str(BOUNDARY),
        "variables": ", ".join(VARIABLES),
    }

    clean["maharashtra_mask"] = mask

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    print("\nSaving master dataset...")

    clean.to_netcdf(
        OUTPUT,
        engine="netcdf4",
    )

    print("\n" + "=" * 70)
    print("MASTER DATASET CREATED")
    print("=" * 70)

    print(f"Output: {OUTPUT}")
    print(f"Size: {OUTPUT.stat().st_size / (1024**3):.2f} GB")

    print("\nDataset dimensions:")

    for name, size in clean.sizes.items():
        print(f"  {name}: {size}")

    print("\nVariables:")

    for variable in clean.data_vars:
        print(f"  - {variable}")

    print("\nPreparation completed successfully.")

    ds.close()


if __name__ == "__main__":
    main()