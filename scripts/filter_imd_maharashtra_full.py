import pandas as pd
import geopandas as gpd
import os

INPUT = r"data\live\imd_2026-09-20_full.csv"
BOUNDARY = r"data\boundary\maharashtra_wgs84.shp"
OUTPUT = r"data\live\imd_maharashtra_2026-09-20_full.csv"

df = pd.read_csv(INPUT)

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(
        df["longitude"],
        df["latitude"]
    ),
    crs="EPSG:4326"
)

maha = gpd.read_file(BOUNDARY)

if maha.crs != "EPSG:4326":
    maha = maha.to_crs("EPSG:4326")

maha_geometry = maha.geometry.union_all()

gdf["inside_maharashtra"] = gdf.geometry.within(
    maha_geometry
)

filtered = gdf[
    gdf["inside_maharashtra"]
].copy()

filtered = filtered.drop(
    columns=["geometry", "inside_maharashtra"],
    errors="ignore"
)

os.makedirs(
    os.path.dirname(OUTPUT),
    exist_ok=True
)

filtered.to_csv(
    OUTPUT,
    index=False
)

print("Original observations:", len(df))
print("Maharashtra observations:", len(filtered))
print("Maharashtra stations:", filtered["station"].nunique())
print("Saved:", OUTPUT)