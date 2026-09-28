import os
import numpy as np

BASE_DIR = r"C:\Users\HP\climate-twin"

INPUT_DIR = os.path.join(BASE_DIR, "data", "live", "imd_processed")
MASK_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed_daily",
    "cnn_dataset",
    "multivariable_prepared",
    "spatial_mask.npy"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "live", "imd_clean")
os.makedirs(OUTPUT_DIR, exist_ok=True)

mask = np.load(MASK_FILE).astype(bool)

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".npy"):
        continue

    data = np.load(os.path.join(INPUT_DIR, filename)).astype(np.float32)

    if data.shape != mask.shape:
        raise ValueError(
            f"{filename}: shape {data.shape}, expected {mask.shape}"
        )

    if np.any(~np.isfinite(data[mask])):
        raise ValueError(
            f"{filename}: missing values inside model mask"
        )

    clean = np.where(mask, data, 0.0).astype(np.float32)

    np.save(
        os.path.join(OUTPUT_DIR, filename),
        clean
    )

    print(
        filename,
        "shape=", clean.shape,
        "valid_mask_cells=", np.sum(mask),
        "invalid_inside_mask=", np.sum(~np.isfinite(clean[mask]))
    )