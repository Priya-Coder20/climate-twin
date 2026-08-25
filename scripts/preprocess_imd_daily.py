import os
import numpy as np
import xarray as xr

BASE = r"data\imd"
OUTPUT_DIR = r"data\processed_daily\imd"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# Official IMD grid definitions
# --------------------------------------------------

TEMP_LAT = np.arange(7.5, 38.5, 1.0)
TEMP_LON = np.arange(67.5, 98.5, 1.0)

RAIN_LAT = np.arange(6.5, 38.75, 0.25)
RAIN_LON = np.arange(66.5, 100.25, 0.25)


def read_temperature(path, year):

    days = 366 if year % 4 == 0 else 365

    data = np.fromfile(path, dtype=np.float32)

    expected = days * 31 * 31

    if len(data) != expected:
        raise ValueError(
            f"{path}: expected {expected} values, "
            f"found {len(data)}"
        )

    data = data.reshape(days, 31, 31)

    # IMD temperature missing value
    data[data == 99.9] = np.nan

    time = np.arange(
        np.datetime64(f"{year}-01-01"),
        np.datetime64(f"{year + 1}-01-01"),
        dtype="datetime64[D]"
    )

    return xr.DataArray(
        data,
        coords={
            "time": time,
            "latitude": TEMP_LAT,
            "longitude": TEMP_LON
        },
        dims=["time", "latitude", "longitude"],
        name="temperature"
    )


def read_rainfall(path, year):

    days = 366 if year % 4 == 0 else 365

    data = np.fromfile(path, dtype=np.float32)

    expected = days * 129 * 135

    if len(data) != expected:
        raise ValueError(
            f"{path}: expected {expected} values, "
            f"found {len(data)}"
        )

    data = data.reshape(days, 129, 135)

    # IMD rainfall missing value
    data[data == -999.0] = np.nan

    time = np.arange(
        np.datetime64(f"{year}-01-01"),
        np.datetime64(f"{year + 1}-01-01"),
        dtype="datetime64[D]"
    )

    return xr.DataArray(
        data,
        coords={
            "time": time,
            "latitude": RAIN_LAT,
            "longitude": RAIN_LON
        },
        dims=["time", "latitude", "longitude"],
        name="rainfall"
    )


# --------------------------------------------------
# MAXIMUM AND MINIMUM TEMPERATURE
# --------------------------------------------------

max_temperature = []
min_temperature = []

for year in range(2021, 2026):

    print(f"Processing temperature: {year}")

    max_path = os.path.join(
        BASE,
        "maxT",
        f"Maxtemp_MaxT_{year}.GRD"
    )

    min_path = os.path.join(
        BASE,
        "minT",
        f"Mintemp_MinT_{year}.GRD"
    )

    max_temperature.append(
        read_temperature(max_path, year)
    )

    min_temperature.append(
        read_temperature(min_path, year)
    )


max_temperature = xr.concat(
    max_temperature,
    dim="time"
)

min_temperature = xr.concat(
    min_temperature,
    dim="time"
)


temperature_ds = xr.Dataset({
    "max_temperature": max_temperature,
    "min_temperature": min_temperature
})

temperature_output = os.path.join(
    OUTPUT_DIR,
    "imd_temperature_daily.nc"
)

temperature_ds.to_netcdf(temperature_output)


# --------------------------------------------------
# RAINFALL
# --------------------------------------------------

rainfall = []

for year in range(2021, 2026):

    print(f"Processing rainfall: {year}")

    rain_path = os.path.join(
        BASE,
        "rainfall",
        f"Rainfall_ind{year}_rfp25.grd"
    )

    rainfall.append(
        read_rainfall(rain_path, year)
    )


rainfall = xr.concat(
    rainfall,
    dim="time"
)


rainfall_ds = xr.Dataset({
    "rainfall": rainfall
})

rainfall_output = os.path.join(
    OUTPUT_DIR,
    "imd_rainfall_daily.nc"
)

rainfall_ds.to_netcdf(rainfall_output)


print()
print("========================================")
print("IMD DAILY PREPROCESSING COMPLETED")
print("========================================")
print("Temperature:", temperature_output)
print("Rainfall:", rainfall_output)