
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent
forecast_dir = root / "outputs" / "live_prediction"
mask_path = (
    root / "data" / "processed_daily" / "cnn_dataset"
    / "multivariable_prepared" / "spatial_mask.npy"
)

mask = np.load(mask_path) > 0
names = ["Tmax", "Tmin", "Rainfall"]

for filename in ["live_forecast.npy", "comparison_forecast.npy"]:
    path = forecast_dir / filename
    a = np.load(path)

    print(f"\n=== {filename} ===")
    print("Shape:", a.shape)
    print("Mask shape:", mask.shape)

    for i, name in enumerate(names):
        x = a[..., i]
        inside = np.broadcast_to(mask, x.shape)
        finite = np.isfinite(x)
        valid = inside & finite

        print(f"\n{name}")
        print("Finite inside mask:", int((finite & inside).sum()))
        print("Missing inside mask:", int((~finite & inside).sum()))
        print("Finite outside mask:", int((finite & ~inside).sum()))
        print("Zeros inside mask:", int(((x == 0) & inside).sum()))

        if valid.any():
            print("Valid minimum:", float(x[valid].min()))
            print("Valid maximum:", float(x[valid].max()))
        else:
            print("No finite values inside mask")
