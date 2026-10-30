
from pathlib import Path
import json
import numpy as np

sections = {
    "src/data/prepare_enhanced_convlstm_data.py": [(1, 120), (350, 440), (490, 545)],
    "src/models/generate_live_forecast.py": [(1, 180)],
}

for filename, ranges in sections.items():
    path = Path(filename)
    print("\n" + "=" * 75)
    print("FILE:", filename)

    if not path.exists():
        print("File not found")
        continue

    lines = path.read_text(encoding="utf-8").splitlines()

    for start, end in ranges:
        print(f"\n--- Lines {start}-{end} ---")
        for number in range(start, min(end, len(lines)) + 1):
            print(f"{number}: {lines[number - 1]}")

print("\n" + "=" * 75)
print("EXISTING LIVE INPUT ARRAY")

array_path = Path("data/live/model_input/X_live.npy")
if array_path.exists():
    arr = np.load(array_path, mmap_mode="r")
    print("Shape:", arr.shape)
    print("Data type:", arr.dtype)
    print("Finite values:", int(np.isfinite(arr).sum()))
    print("Total values:", arr.size)

metadata_path = Path("data/live/model_input/X_live_metadata.json")
if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    print("Features:", metadata.get("features"))
    print("Input dates:", metadata.get("input_dates"))
