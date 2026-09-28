import os
import numpy as np
import xarray as xr

BASE_DIR = r"C:\Users\HP\climate-twin"
INPUT_DIR = os.path.join(BASE_DIR, "data", "live", "era5")
OUTPUT_DIR = os.path.join(INPUT_DIR, "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_LAT = np.linspace(15.63, 22.02, 65)
TARGET_LON = np.linspace(72.65, 79.84, 73)


def regrid(da):
    return (
        da.interp(
            latitude=TARGET_LAT,
            longitude=TARGET_LON,
            method="linear"
        )
        .interpolate_na(
            dim="latitude",
            method="linear",
            fill_value="extrapolate"
        )
        .interpolate_na(
            dim="longitude",
            method="linear",
            fill_value="extrapolate"
        )
    )


for date in sorted(os.listdir(INPUT_DIR)):

    folder = os.path.join(INPUT_DIR, date)

    if not os.path.isdir(folder):
        continue

    instant_file = os.path.join(
        folder,
        "data_stream-oper_stepType-instant.nc"
    )

    accum_file = os.path.join(
        folder,
        "data_stream-oper_stepType-accum.nc"
    )

    if not os.path.exists(instant_file):
        continue

    if not os.path.exists(accum_file):
        continue

    print("Processing:", date)

    instant = xr.open_dataset(instant_file)
    accum = xr.open_dataset(accum_file)

    temperature = regrid(instant["t2m"]).mean("valid_time") - 273.15
    dewpoint = regrid(instant["d2m"]).mean("valid_time") - 273.15
    pressure = regrid(instant["msl"]).mean("valid_time") / 100.0

    u10 = regrid(instant["u10"])
    v10 = regrid(instant["v10"])

    wind_speed = np.sqrt(
        u10**2 + v10**2
    ).mean("valid_time")

    rainfall = regrid(accum["tp"]).sum("valid_time") * 1000.0
    rainfall = rainfall.clip(min=0)

    variables = {
        "era5_temperature": temperature,
        "era5_dewpoint": dewpoint,
        "era5_pressure": pressure,
        "era5_wind_speed": wind_speed,
        "era5_rainfall": rainfall,
    }

    for name, data in variables.items():

        arr = np.asarray(
            data.values,
            dtype=np.float32
        )

        if arr.shape != (65, 73):
            raise ValueError(
                f"{name} {date}: shape {arr.shape}"
            )

        if not np.isfinite(arr).all():
            raise ValueError(
                f"{name} {date}: invalid values"
            )

        np.save(
            os.path.join(
                OUTPUT_DIR,
                f"{name}_{date}.npy"
            ),
            arr
        )

        print(
            name,
            "min=", round(float(arr.min()), 4),
            "max=", round(float(arr.max()), 4),
            "mean=", round(float(arr.mean()), 4)
        )

    instant.close()
    accum.close()

print("ERA5 preprocessing completed.")