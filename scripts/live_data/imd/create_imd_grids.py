import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

BASE_DIR = r"C:\Users\HP\climate-twin"

INPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "live",
    "imd_historical_daily"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "live",
    "imd_processed"
)

MASK_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed_daily",
    "cnn_dataset",
    "multivariable_prepared",
    "spatial_mask.npy"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

mask = np.load(MASK_FILE).astype(bool)

target_lat = np.linspace(15.63, 22.02, 65)
target_lon = np.linspace(72.65, 79.84, 73)

lon_grid, lat_grid = np.meshgrid(
    target_lon,
    target_lat
)

variables = ["tmax", "tmin", "rainfall"]

files = sorted(
    glob.glob(
        os.path.join(
            INPUT_DIR,
            "imd_daily_2026-09-*.csv"
        )
    )
)

for file in files:

    date = os.path.basename(file)[10:20]

    print("\nProcessing:", date)

    df = pd.read_csv(file)

    for variable in variables:

        valid = df[
            df[variable].notna()
        ].copy()

        if len(valid) < 3:
            raise ValueError(
                f"{date} {variable}: insufficient stations"
            )

        points = valid[
            ["longitude", "latitude"]
        ].values

        values = valid[
            variable
        ].values

        grid = griddata(
            points,
            values,
            (lon_grid, lat_grid),
            method="linear"
        )

        missing = ~np.isfinite(grid)

        if np.any(missing):

            nearest = griddata(
                points,
                values,
                (lon_grid, lat_grid),
                method="nearest"
            )

            grid[missing] = nearest[missing]

        grid = grid.astype(np.float32)

        grid[~mask] = 0.0

        output = os.path.join(
            OUTPUT_DIR,
            f"{variable}_{date}.npy"
        )

        np.save(
            output,
            grid
        )

        valid_inside = grid[mask]

        print(
            variable,
            "shape=",
            grid.shape,
            "min=",
            round(float(valid_inside.min()), 3),
            "max=",
            round(float(valid_inside.max()), 3),
            "mean=",
            round(float(valid_inside.mean()), 3),
            "NaN inside mask=",
            np.sum(~np.isfinite(valid_inside))
        )

print("\nIMD grid generation completed.")