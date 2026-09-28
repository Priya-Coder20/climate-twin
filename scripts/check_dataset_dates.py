import xarray as xr
import numpy as np

FILE = r"data\processed_daily\cnn_dataset\common_daily_v2.nc"

ds = xr.open_dataset(FILE)

dates = ds.time.values.astype("datetime64[D]")

print("Number of dates:", len(dates))
print("First date:", dates[0])
print("Last date:", dates[-1])

# Differences between consecutive dates
diffs = np.diff(dates).astype("timedelta64[D]").astype(int)

print("\nDate gap statistics:")
print("Minimum gap:", diffs.min(), "days")
print("Maximum gap:", diffs.max(), "days")
print("Average gap:", diffs.mean(), "days")

# Find gaps larger than one day
gap_indices = np.where(diffs > 1)[0]

print("\nNumber of gaps > 1 day:", len(gap_indices))

if len(gap_indices) > 0:
    print("\nMissing-date gaps:")

    for i in gap_indices[:20]:
        print(
            dates[i],
            "->",
            dates[i + 1],
            "| gap:",
            diffs[i],
            "days"
        )

# Show split boundaries based on sequence count
WINDOW = 7
n_samples = len(dates) - WINDOW

train_end = int(n_samples * 0.70)
val_end = train_end + int(n_samples * 0.15)

print("\nSequence count:", n_samples)

print("\nApproximate sequence split:")
print("Train sequences:", train_end)
print("Validation sequences:", val_end - train_end)
print("Test sequences:", n_samples - val_end)

# Each sequence predicts date i + WINDOW
print("\nApproximate target-date boundaries:")

print(
    "Train target dates:",
    dates[WINDOW],
    "->",
    dates[train_end + WINDOW - 1]
)

print(
    "Validation target dates:",
    dates[train_end + WINDOW],
    "->",
    dates[val_end + WINDOW - 1]
)

print(
    "Test target dates:",
    dates[val_end + WINDOW],
    "->",
    dates[-1]
)

ds.close()