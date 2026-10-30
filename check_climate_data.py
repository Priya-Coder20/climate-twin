
from pathlib import Path
import numpy as np

ROOT = Path("data/live/imd_clean")

if not ROOT.exists():
    raise SystemExit(f"Data folder not found: {ROOT.resolve()}")

files = sorted(ROOT.glob("*.npy"))

if not files:
    raise SystemExit(f"No .npy files found in {ROOT.resolve()}")

for path in files:
    print(f"\n{'=' * 60}")
    print(f"File: {path}")

    try:
        arr = np.load(path, allow_pickle=False)

        print("Shape:", arr.shape)
        print("Data type:", arr.dtype)

        if not np.issubdtype(arr.dtype, np.number):
            print("WARNING: Non-numeric array")
            continue

        finite = np.isfinite(arr)
        valid = arr[finite]

        print("Total values:", arr.size)
        print("Finite values:", valid.size)
        print("NaN/Inf values:", arr.size - valid.size)

        if valid.size == 0:
            print("WARNING: No finite numeric values")
            continue

        print("Minimum:", float(valid.min()))
        print("Maximum:", float(valid.max()))
        print("Mean:", float(valid.mean()))
        print("Median:", float(np.median(valid)))
        print("Zero fraction:", float(np.mean(valid == 0)))
        print("Percentiles (5, 50, 95):",
              np.percentile(valid, [5, 50, 95]).tolist())

    except Exception as exc:
        print("ERROR:", repr(exc))
