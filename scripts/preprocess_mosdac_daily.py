import os
import glob
import re
from datetime import datetime

import rasterio
import geopandas as gpd
import numpy as np

from rasterio.mask import mask
from rasterio.warp import reproject, Resampling


# PROJECT PATHS
PROJECT = r"C:\Users\HP\climate-twin"

IMR_DIR = os.path.join(PROJECT, r"data\mosdac\imr")
PET_DIR = os.path.join(PROJECT, r"data\mosdac\pet")

BOUNDARY_FILE = os.path.join(
    PROJECT,
    r"data\boundary\maharashtra_wgs84.shp"
)

OUTPUT_DIR = os.path.join(
    PROJECT,
    r"data\processed_daily\combined"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# SETTINGS
MAX_DAYS = None


# EXTRACT DATE FROM MOSDAC FILENAME
def extract_date(filename):

    match = re.search(
        r"_(\d{2}[A-Z]{3}\d{4})_",
        os.path.basename(filename)
    )

    if match is None:
        return None

    return datetime.strptime(
        match.group(1),
        "%d%b%Y"
    ).strftime("%Y-%m-%d")

# FIND IMR AND PET FILES
imr_files = glob.glob(
    os.path.join(IMR_DIR, "*.tif")
)

pet_files = glob.glob(
    os.path.join(PET_DIR, "*.tif")
)


imr_by_date = {}

for file in imr_files:

    date = extract_date(file)

    if date:
        imr_by_date[date] = file


pet_by_date = {}

for file in pet_files:

    date = extract_date(file)

    if date:
        pet_by_date[date] = file


# FIND COMMON DATES
common_dates = sorted(
    set(imr_by_date.keys()) &
    set(pet_by_date.keys())
)

dates_to_process = common_dates if MAX_DAYS is None else common_dates[:MAX_DAYS]


# LOAD MAHARASHTRA BOUNDARY
boundary = gpd.read_file(BOUNDARY_FILE)


# PROCESS EACH DAY
for date in dates_to_process:

    imr_file = imr_by_date[date]
    pet_file = pet_by_date[date]

    output_folder = os.path.join(
        OUTPUT_DIR,
        date
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    
    # READ AND CLIP IMR
    with rasterio.open(imr_file) as imr:

        boundary_imr = boundary.to_crs(imr.crs)

        imr_data, imr_transform = mask(
            imr,
            boundary_imr.geometry,
            crop=True,
            nodata=-999.0
        )

        imr_data = imr_data[0]

        profile = imr.profile.copy()

        profile.update(
            height=imr_data.shape[0],
            width=imr_data.shape[1],
            transform=imr_transform,
            crs=imr.crs,
            dtype="float32",
            nodata=-999.0,
            compress="deflate"
        )

    # REPROJECT AND RESAMPLE PET
    with rasterio.open(pet_file) as pet:

        pet_data = pet.read(1)

        pet_data = np.where(
            pet_data == pet.nodata,
            np.nan,
            pet_data
        )

        pet_aligned = np.full(
            imr_data.shape,
            np.nan,
            dtype=np.float32
        )

        reproject(
            source=pet_data,
            destination=pet_aligned,
            src_transform=pet.transform,
            src_crs=pet.crs,
            dst_transform=imr_transform,
            dst_crs=imr.crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan
        )

        pet_aligned = np.where(
            np.isnan(pet_aligned),
            -999.0,
            pet_aligned
        ).astype(np.float32)


    # SAVE IMR
    imr_output = os.path.join(
        output_folder,
        "imr.tif"
    )

    with rasterio.open(
        imr_output,
        "w",
        **profile
    ) as dst:

        dst.write(
            imr_data.astype(np.float32),
            1
        )

    # SAVE PET
    pet_output = os.path.join(
        output_folder,
        "pet_aligned.tif"
    )

    with rasterio.open(
        pet_output,
        "w",
        **profile
    ) as dst:

        dst.write(
            pet_aligned,
            1
        )


print("Processed", len(dates_to_process), "common IMR/PET days.")
print("Output:", OUTPUT_DIR)