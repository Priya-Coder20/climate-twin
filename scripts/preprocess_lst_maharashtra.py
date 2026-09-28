import os
import glob
import re
from datetime import datetime

import numpy as np
import rasterio
import geopandas as gpd
from rasterio.mask import mask


# PATHS
PROJECT = r"C:\Users\HP\climate-twin"

LST_DIR = os.path.join(
    PROJECT,
    r"data\mosdac\lst"
)

BOUNDARY_FILE = os.path.join(
    PROJECT,
    r"data\processed_daily\maharashtra_boundary_wgs84.shp"
)

OUTPUT_DIR = os.path.join(
    PROJECT,
    r"data\processed_daily\lst_maharashtra"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# EXTRACT DATE
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


# FIND LST FILES
lst_files = glob.glob(
    os.path.join(LST_DIR, "*.tif")
)

print("LST files found:", len(lst_files))



# LOAD MAHARASHTRA BOUNDARY
boundary = gpd.read_file(
    BOUNDARY_FILE
)


# GROUP FILES BY DATE
files_by_date = {}

for file in lst_files:

    date = extract_date(file)

    if date is not None:

        files_by_date.setdefault(
            date,
            []
        ).append(file)


print("Unique LST dates:", len(files_by_date))


# PROCESS EACH DATE
for date in sorted(files_by_date):

    files = files_by_date[date]

    print("Processing:", date)

    daily_arrays = []

    output_transform = None
    output_profile = None

    for file in files:

        with rasterio.open(file) as src:

            # Reproject boundary to raster CRS
            boundary_raster = boundary.to_crs(
                src.crs
            )

            # Clip raster to Maharashtra
            data, transform = mask(
                src,
                boundary_raster.geometry,
                crop=True,
                nodata=-999.0
            )

            data = data[0].astype(
                np.float32
            )

            # Convert invalid values to NaN
            data[data == -999.0] = np.nan

            daily_arrays.append(data)

            output_transform = transform

            output_profile = src.profile.copy()


    # Combine observations from same date
    stacked = np.stack(
        daily_arrays,
        axis=0
    )

    daily_lst = np.nanmean(
        stacked,
        axis=0
    ).astype(np.float32)


    # Save
    output_profile.update(
        driver="GTiff",
        height=daily_lst.shape[0],
        width=daily_lst.shape[1],
        transform=output_transform,
        dtype="float32",
        count=1,
        nodata=-999.0,
        compress="deflate"
    )

    daily_lst = np.where(
        np.isnan(daily_lst),
        -999.0,
        daily_lst
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"LST_{date}.tif"
    )

    with rasterio.open(
        output_file,
        "w",
        **output_profile
    ) as dst:

        dst.write(
            daily_lst,
            1
        )


# COMPLETE
print()
print("MOSDAC LST PREPROCESSING COMPLETED")
print("Input files:", len(lst_files))
print("Unique dates:", len(files_by_date))
print("Output:", OUTPUT_DIR)