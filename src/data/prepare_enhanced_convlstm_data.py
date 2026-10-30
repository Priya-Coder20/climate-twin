import os
import numpy as np
import xarray as xr


DATA_PATH = r"data\processed\master_climate_dataset.nc"
OUTPUT_DIR = r"data\processed\enhanced_convlstm_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)


FEATURES = [
    "imr",
    "pet",
    "era5_temperature",
    "era5_dewpoint",
    "era5_wind_speed",
    "era5_pressure",
    "era5_rainfall",
    "imd_max_temperature",
    "imd_min_temperature",
    "imd_rainfall",
]

TARGETS = [
    "imd_max_temperature",
    "imd_min_temperature",
    "imd_rainfall",
]

INPUT_DAYS = 7
FORECAST_DAYS = 7
MIN_COVERAGE = 0.80


print("Loading master dataset...")

ds = xr.open_dataset(DATA_PATH)

mask = ds["maharashtra_mask"].values.astype(bool)
dates = ds.time.values


# ---------------------------------------------------------
# Load original climate features
# ---------------------------------------------------------

features = np.stack(
    [ds[v].values for v in FEATURES],
    axis=-1
)

targets = np.stack(
    [ds[v].values for v in TARGETS],
    axis=-1
)


# ---------------------------------------------------------
# Add seasonal features
# ---------------------------------------------------------

print("Creating seasonal features...")

day_of_year = (
    ds.time.dt.dayofyear.values.astype(np.float32)
)

# Handle leap-year day count consistently.
year_length = np.where(
    ds.time.dt.is_leap_year.values,
    366.0,
    365.0
)

season_angle = (
    2.0 * np.pi *
    (day_of_year - 1.0) /
    year_length
)

sin_season = np.sin(season_angle)
cos_season = np.cos(season_angle)


n_time = len(dates)
n_lat = len(ds.latitude)
n_lon = len(ds.longitude)

season_features = np.zeros(
    (n_time, n_lat, n_lon, 2),
    dtype=np.float32
)

season_features[..., 0] = (
    sin_season[:, None, None]
)

season_features[..., 1] = (
    cos_season[:, None, None]
)


# ---------------------------------------------------------
# Calculate rainfall accumulation features
# ---------------------------------------------------------

print("Creating rainfall accumulation features...")

rainfall = ds["imd_rainfall"].values.astype(np.float32)


rainfall_3day = np.full_like(
    rainfall,
    np.nan,
    dtype=np.float32
)

rainfall_7day = np.full_like(
    rainfall,
    np.nan,
    dtype=np.float32
)


for i in range(n_time):

    if i >= 2:
        window = rainfall[i - 2:i + 1]

        if np.isfinite(window).all(axis=0).any():
            rainfall_3day[i] = np.nansum(
                window,
                axis=0
            )

    if i >= 6:
        window = rainfall[i - 6:i + 1]

        if np.isfinite(window).all(axis=0).any():
            rainfall_7day[i] = np.nansum(
                window,
                axis=0
            )


rainfall_accumulation = np.stack(
    [
        rainfall_3day,
        rainfall_7day
    ],
    axis=-1
)


# ---------------------------------------------------------
# Combine all features
# ---------------------------------------------------------

features = np.concatenate(
    [
        features,
        season_features,
        rainfall_accumulation
    ],
    axis=-1
)


FEATURE_NAMES = FEATURES + [
    "sin_season",
    "cos_season",
    "imd_rainfall_3day",
    "imd_rainfall_7day",
]


print()
print("Features:", len(FEATURE_NAMES))
print(FEATURE_NAMES)


# ---------------------------------------------------------
# Determine usable spatial cells
# ---------------------------------------------------------

feature_valid = np.isfinite(
    features
).all(axis=-1)

target_valid = np.isfinite(
    targets
).all(axis=-1)

combined_valid = (
    feature_valid &
    target_valid
)

coverage = combined_valid.mean(axis=0)

usable_cells = (
    mask &
    (coverage >= MIN_COVERAGE)
)


print()
print(
    "Usable cells:",
    int(usable_cells.sum())
)


# ---------------------------------------------------------
# Create consecutive 7 + 7 day sequences
# ---------------------------------------------------------

X = []
Y = []

print()
print("Creating sequences...")


for i in range(
    len(dates) -
    INPUT_DAYS -
    FORECAST_DAYS +
    1
):

    window_dates = dates[
        i:i + INPUT_DAYS + FORECAST_DAYS
    ]

    # Require true consecutive calendar days.
    if not np.all(
        np.diff(window_dates) ==
        np.timedelta64(1, "D")
    ):
        continue


    x = features[
        i:i + INPUT_DAYS
    ].copy()


    y = targets[
        i + INPUT_DAYS:
        i + INPUT_DAYS + FORECAST_DAYS
    ].copy()


    # Only keep sequences with complete data
    # over the usable Maharashtra cells.
    if not np.isfinite(
        x[:, usable_cells, :]
    ).all():
        continue


    if not np.isfinite(
        y[:, usable_cells, :]
    ).all():
        continue


    # Outside Maharashtra = zero.
    x[:, ~mask, :] = 0
    y[:, ~mask, :] = 0


    X.append(x)
    Y.append(y)


if not X:
    raise RuntimeError(
        "No valid enhanced ConvLSTM sequences were created."
    )


X = np.asarray(
    X,
    dtype=np.float32
)

Y = np.asarray(
    Y,
    dtype=np.float32
)


print()
print("All sequences:", X.shape)


# ---------------------------------------------------------
# Chronological train / validation / test split
# ---------------------------------------------------------

n = len(X)

train_end = int(n * 0.70)
val_end = int(n * 0.85)


X_train = X[:train_end]
Y_train = Y[:train_end]

X_val = X[train_end:val_end]
Y_val = Y[train_end:val_end]

X_test = X[val_end:]
Y_test = Y[val_end:]


print()
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_val:", X_val.shape)
print("Y_val:", Y_val.shape)
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)


# ---------------------------------------------------------
# Feature normalization
# ---------------------------------------------------------

train_values = X_train[
    :,
    :,
    usable_cells,
    :
].reshape(
    -1,
    len(FEATURE_NAMES)
)


feature_mean = np.nanmean(
    train_values,
    axis=0
)

feature_std = np.nanstd(
    train_values,
    axis=0
)


# ---------------------------------------------------------
# Target normalization
# ---------------------------------------------------------

target_values = Y_train[
    :,
    :,
    usable_cells,
    :
].reshape(
    -1,
    len(TARGETS)
)


target_mean = np.nanmean(
    target_values,
    axis=0
)

target_std = np.nanstd(
    target_values,
    axis=0
)


feature_std[
    feature_std == 0
] = 1

target_std[
    target_std == 0
] = 1


# ---------------------------------------------------------
# Normalize
# ---------------------------------------------------------

X_train = (
    X_train -
    feature_mean
) / feature_std

X_val = (
    X_val -
    feature_mean
) / feature_std

X_test = (
    X_test -
    feature_mean
) / feature_std


Y_train = (
    Y_train -
    target_mean
) / target_std

Y_val = (
    Y_val -
    target_mean
) / target_std

Y_test = (
    Y_test -
    target_mean
) / target_std


# ---------------------------------------------------------
# Remove any remaining non-finite values
# ---------------------------------------------------------

X_train = np.nan_to_num(
    X_train,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

X_val = np.nan_to_num(
    X_val,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

X_test = np.nan_to_num(
    X_test,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


Y_train = np.nan_to_num(
    Y_train,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

Y_val = np.nan_to_num(
    Y_val,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

Y_test = np.nan_to_num(
    Y_test,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# ---------------------------------------------------------
# Save everything
# ---------------------------------------------------------

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_train.npy"
    ),
    X_train
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "Y_train.npy"
    ),
    Y_train
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_val.npy"
    ),
    X_val
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "Y_val.npy"
    ),
    Y_val
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_test.npy"
    ),
    X_test
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "Y_test.npy"
    ),
    Y_test
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "feature_mean.npy"
    ),
    feature_mean
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "feature_std.npy"
    ),
    feature_std
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "target_mean.npy"
    ),
    target_mean
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "target_std.npy"
    ),
    target_std
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "maharashtra_mask.npy"
    ),
    mask
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "usable_cells.npy"
    ),
    usable_cells
)


# Save feature names for reference.
with open(
    os.path.join(
        OUTPUT_DIR,
        "feature_names.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    for name in FEATURE_NAMES:
        f.write(name + "\n")


ds.close()


print()
print("Enhanced ConvLSTM data preparation complete.")
print("Output:", OUTPUT_DIR)
print("X finite:", np.isfinite(X_train).all())
print("Y finite:", np.isfinite(Y_train).all())