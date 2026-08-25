import os
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point

# --------------------------------------------------
# Paths
# --------------------------------------------------

TEMP_INPUT = r"data\processed_daily\imd\imd_temperature_daily.nc"
RAIN_INPUT = r"data\processed_daily\imd\imd_rainfall_daily.nc"

BOUNDARY = r"data\processed_daily\maharashtra_boundary_wgs84.shp"

OUTPUT_DIR = r"data\processed_daily\imd_maharashtra"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Load Maharashtra boundary
# --------------------------------------------------

print("Loading Maharashtra boundary...")

maharashtra = gpd.read_file(BOUNDARY)
maharashtra = maharashtra.to_crs("EPSG:4326")

geometry = maharashtra.geometry.union_all()

minx, miny, maxx, maxy = geometry.bounds

print("Maharashtra bounds:")
print("Longitude:", minx, "to", maxx)
print("Latitude :", miny, "to", maxy)


# --------------------------------------------------
# Create polygon mask
# --------------------------------------------------

def create_mask(latitudes, longitudes):

    mask = np.zeros(
        (len(latitudes), len(longitudes)),
        dtype=bool
    )

    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):

            point = Point(float(lon), float(lat))

            if geometry.contains(point) or geometry.touches(point):
                mask[i, j] = True

    return mask


# --------------------------------------------------
# Process temperature
# --------------------------------------------------

print()
print("Processing IMD temperature...")

temp_ds = xr.open_dataset(TEMP_INPUT)

# First crop to Maharashtra bounding box
temp_ds = temp_ds.sel(
    latitude=slice(miny, maxy),
    longitude=slice(minx, maxx)
)

# Create exact Maharashtra polygon mask
temp_mask = create_mask(
    temp_ds.latitude.values,
    temp_ds.longitude.values
)

temp_mask_da = xr.DataArray(
    temp_mask,
    coords={
        "latitude": temp_ds.latitude,
        "longitude": temp_ds.longitude
    },
    dims=["latitude", "longitude"]
)

# Mask pixels outside Maharashtra
temp_maharashtra = temp_ds.where(temp_mask_da)

temp_output = os.path.join(
    OUTPUT_DIR,
    "imd_temperature_maharashtra_daily.nc"
)

temp_maharashtra.to_netcdf(temp_output)

print("Temperature saved:")
print(temp_output)

temp_ds.close()


# --------------------------------------------------
# Process rainfall
# --------------------------------------------------

print()
print("Processing IMD rainfall...")

rain_ds = xr.open_dataset(RAIN_INPUT)

# Crop to Maharashtra bounding box
rain_ds = rain_ds.sel(
    latitude=slice(miny, maxy),
    longitude=slice(minx, maxx)
)

# Create exact Maharashtra polygon mask
rain_mask = create_mask(
    rain_ds.latitude.values,
    rain_ds.longitude.values
)

rain_mask_da = xr.DataArray(
    rain_mask,
    coords={
        "latitude": rain_ds.latitude,
        "longitude": rain_ds.longitude
    },
    dims=["latitude", "longitude"]
)

# Mask pixels outside Maharashtra
rain_maharashtra = rain_ds.where(rain_mask_da)

rain_output = os.path.join(
    OUTPUT_DIR,
    "imd_rainfall_maharashtra_daily.nc"
)

rain_maharashtra.to_netcdf(rain_output)

print("Rainfall saved:")
print(rain_output)

rain_ds.close()


# --------------------------------------------------
# Finished
# --------------------------------------------------

print()
print("========================================")
print("MAHARASHTRA IMD SUBSETTING COMPLETED")
print("========================================")