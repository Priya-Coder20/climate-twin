import os
import glob
import re
import numpy as np
import h5py
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

BASE_DIR = r"C:\Users\HP\climate-twin"

INPUT_DIR = os.path.join(BASE_DIR, "data", "live", "mosdac")
OUTPUT_DIR = os.path.join(INPUT_DIR, "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_LAT = np.linspace(15.63, 22.02, 65)
TARGET_LON = np.linspace(72.65, 79.84, 73)

MONTHS = {
    "JAN": "01",
    "FEB": "02",
    "MAR": "03",
    "APR": "04",
    "MAY": "05",
    "JUN": "06",
    "JUL": "07",
    "AUG": "08",
    "SEP": "09",
    "OCT": "10",
    "NOV": "11",
    "DEC": "12"
}


def get_date(filepath):
    filename = os.path.basename(filepath)

    match = re.search(
        r"3RIMG_(\d{2})([A-Z]{3})(\d{4})",
        filename
    )

    if not match:
        return None

    day, month, year = match.groups()

    return f"{year}-{MONTHS[month]}-{day}"


def process_imr(filepath):
    with h5py.File(filepath, "r") as f:
        data = f["IMR_DLY"][0]
        lat = f["latitude"][:]
        lon = f["longitude"][:]

    interpolator = RegularGridInterpolator(
        (lat, lon),
        data,
        method="linear",
        bounds_error=False,
        fill_value=0.0
    )

    target_lat, target_lon = np.meshgrid(
        TARGET_LAT,
        TARGET_LON,
        indexing="ij"
    )

    points = np.column_stack(
        (
            target_lat.ravel(),
            target_lon.ravel()
        )
    )

    result = interpolator(points).reshape(65, 73)

    result = np.nan_to_num(
        result,
        nan=0.0
    )

    return result.astype(np.float32)


def process_pet(filepath):
    with h5py.File(filepath, "r") as f:
        data = f["PET_DLY"][0]
        x = f["X"][:]
        y = f["Y"][:]

    data = np.where(
        np.isfinite(data) & (data > -900),
        data,
        np.nan
    )

    transformer = Transformer.from_crs(
        "+proj=longlat +datum=WGS84",
        "+proj=merc +lon_0=77.25 +lat_ts=17.75 +datum=WGS84 +units=m",
        always_xy=True
    )

    target_lon, target_lat = np.meshgrid(
        TARGET_LON,
        TARGET_LAT
    )

    target_x, target_y = transformer.transform(
        target_lon,
        target_lat
    )

    interpolator = RegularGridInterpolator(
        (y, x),
        data,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    points = np.column_stack(
        (
            target_y.ravel(),
            target_x.ravel()
        )
    )

    result = interpolator(points).reshape(65, 73)

    missing = ~np.isfinite(result)

    if np.any(missing):
        nearest = RegularGridInterpolator(
            (y, x),
            data,
            method="nearest",
            bounds_error=False,
            fill_value=np.nan
        )

        nearest_result = nearest(points).reshape(65, 73)

        result = np.where(
            missing,
            nearest_result,
            result
        )

    result = np.where(
        np.isfinite(result),
        result,
        0.0
    )

    return result.astype(np.float32)


imr_files = sorted(
    glob.glob(
        os.path.join(
            INPUT_DIR,
            "3RIMG_*_L3G_IMR_DLY_*.h5"
        )
    )
)

pet_files = sorted(
    glob.glob(
        os.path.join(
            INPUT_DIR,
            "3RIMG_*_L3C_PET_DLY_*.h5"
        )
    )
)

print("MOSDAC → MODEL GRID PREPROCESSING")

print()
print("PROCESSING IMR")

for filepath in imr_files:
    date = get_date(filepath)

    if date is None:
        continue

    print(f"Processing IMR {date}...")

    result = process_imr(filepath)

    output = os.path.join(
        OUTPUT_DIR,
        f"imr_{date}.npy"
    )

    np.save(output, result)

    print(
        f"Saved IMR {date}: "
        f"shape={result.shape}, "
        f"min={result.min():.4f}, "
        f"max={result.max():.4f}, "
        f"mean={result.mean():.4f}"
    )

print()
print("PROCESSING PET")

for filepath in pet_files:
    date = get_date(filepath)

    if date is None:
        continue

    print(f"Processing PET {date}...")

    result = process_pet(filepath)

    output = os.path.join(
        OUTPUT_DIR,
        f"pet_{date}.npy"
    )

    np.save(output, result)

    print(
        f"Saved PET {date}: "
        f"shape={result.shape}, "
        f"min={result.min():.4f}, "
        f"max={result.max():.4f}, "
        f"mean={result.mean():.4f}"
    )

print()
print("PROCESSING COMPLETED")
