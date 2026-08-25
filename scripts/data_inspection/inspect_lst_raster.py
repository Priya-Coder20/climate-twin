import rasterio
import os
import numpy as np

# ============================================================
# RAW MOSDAC LST RASTER INSPECTION
# ============================================================

input_file = (
    r"C:\Users\HP\climate-twin\data\mosdac\lst"
    r"\3RIMG_01OCT2021_0528_L2B_LST_V01R00_LST.tif"
)

print("=" * 70)
print("RAW MOSDAC LST RASTER INSPECTION")
print("=" * 70)

print("\nInput file:")
print(input_file)

# ------------------------------------------------------------
# Check file
# ------------------------------------------------------------

if not os.path.exists(input_file):
    print("\nERROR: File not found.")
    raise SystemExit

# ------------------------------------------------------------
# Open raster
# ------------------------------------------------------------

with rasterio.open(input_file) as src:

    print("\n--- Raster Metadata ---")

    print("Driver          :", src.driver)
    print("CRS             :", src.crs)
    print("Width           :", src.width)
    print("Height          :", src.height)
    print("Number of bands :", src.count)

    print("\n--- Resolution ---")
    print("X resolution    :", src.res[0])
    print("Y resolution    :", src.res[1])

    print("\n--- Bounds ---")
    print("Left            :", src.bounds.left)
    print("Bottom          :", src.bounds.bottom)
    print("Right           :", src.bounds.right)
    print("Top             :", src.bounds.top)

    print("\n--- NoData ---")
    print("NoData value    :", src.nodata)

    print("\n--- Data Type ---")
    print("Data type       :", src.dtypes[0])

    # --------------------------------------------------------
    # Read raster
    # --------------------------------------------------------

    data = src.read(1)

    print("\n--- Raw Data Statistics ---")

    print("Total pixels    :", data.size)

    # Determine invalid pixels
    if src.nodata is not None:
        invalid = data == src.nodata
    else:
        invalid = np.isnan(data)

    valid = ~invalid

    print("Valid pixels    :", np.sum(valid))
    print("Invalid pixels  :", np.sum(invalid))

    if np.sum(valid) > 0:

        valid_data = data[valid]

        print("\nMinimum value   :", np.min(valid_data))
        print("Maximum value   :", np.max(valid_data))
        print("Mean value      :", np.mean(valid_data))
        print("Median value    :", np.median(valid_data))

        print("\nPercentiles:")
        print("1%              :", np.percentile(valid_data, 1))
        print("5%              :", np.percentile(valid_data, 5))
        print("95%             :", np.percentile(valid_data, 95))
        print("99%             :", np.percentile(valid_data, 99))

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)