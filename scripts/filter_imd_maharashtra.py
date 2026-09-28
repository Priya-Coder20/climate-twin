import pandas as pd
import geopandas as gpd
import os

INPUT = r"data\live\imd_daily_2026-09-20.csv"
BOUNDARY = r"data\boundary\maharashtra_wgs84.shp"
OUTPUT = r"data\live\imd_maharashtra_2026-09-20.csv"

print("Loading IMD observations...")
df = pd.read_csv(INPUT)

print("Input rows:", len(df))

# Convert station coordinates to GeoDataFrame
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(
        df["longitude"],
        df["latitude"]
    ),
    crs="EPSG:4326"
)

print("Loading Maharashtra boundary...")
maha = gpd.read_file(BOUNDARY)

print("Boundary CRS:", maha.crs)

# Make sure boundary uses WGS84
if maha.crs != "EPSG:4326":
    maha = maha.to_crs("EPSG:4326")

# Combine all Maharashtra polygons into one geometry
maha_geometry = maha.geometry.union_all()

# Keep points inside Maharashtra
gdf["inside_maharashtra"] = gdf.geometry.within(maha_geometry)

filtered = gdf[gdf["inside_maharashtra"]].copy()

# Remove geometry column before saving CSV
filtered = filtered.drop(
    columns=["geometry", "inside_maharashtra"],
    errors="ignore"
)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

filtered.to_csv(OUTPUT, index=False)

print()
print("Filtering completed.")
print("Original rows:", len(df))
print("Maharashtra rows:", len(filtered))
print("Stations inside Maharashtra:", filtered["station"].nunique())

print()
print("Maharashtra stations:")
print(
    filtered[
        ["station", "latitude", "longitude"]
    ].drop_duplicates().to_string(index=False)
)

print()
print("Saved:", OUTPUT)