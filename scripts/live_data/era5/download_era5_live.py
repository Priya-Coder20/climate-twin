import os
import sys
from datetime import date, timedelta

import cdsapi

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "live",
    "era5"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "live",
    "era5",
    "processed"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_existing_processed_dates():
    if not os.path.exists(PROCESSED_DIR):
        return set()

    dates = set()
    prefix = "era5_temperature_"

    for filename in os.listdir(PROCESSED_DIR):
        if filename.startswith(prefix) and filename.endswith(".npy"):
            date_text = filename[len(prefix):-4]

            try:
                date.fromisoformat(date_text)
                dates.add(date_text)
            except ValueError:
                continue

    return dates


def get_download_dates():
    existing = get_existing_processed_dates()

    if existing:
        latest = max(
            date.fromisoformat(d)
            for d in existing
        )
        start_date = latest + timedelta(days=1)
    else:
        start_date = date.today() - timedelta(days=6)

    end_date = date.today()

    if start_date > end_date:
        return []

    dates = []
    current = start_date

    while current <= end_date:
        dates.append(
            current.strftime("%Y-%m-%d")
        )
        current += timedelta(days=1)

    return dates


def download_day(client, date_text):
    year, month, day = date_text.split("-")

    output_path = os.path.join(
        OUTPUT_DIR,
        date_text
    )

    os.makedirs(
        output_path,
        exist_ok=True
    )

    request = {
        "product_type": "reanalysis",
        "variable": [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "mean_sea_level_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "total_precipitation"
        ],
        "year": year,
        "month": month,
        "day": day,
        "time": [
            "00:00",
            "06:00",
            "12:00",
            "18:00"
        ],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [
            22.1,
            72.6,
            15.5,
            79.9
        ]
    }

    target = os.path.join(
        output_path,
        "era5.nc"
    )

    print(f"Downloading ERA5: {date_text}")

    client.retrieve(
        "reanalysis-era5-single-levels",
        request,
        target
    )


def main():
    dates = get_download_dates()

    if not dates:
        print("ERA5 data is already up to date.")
        return

    client = cdsapi.Client()

    for date_text in dates:
        try:
            download_day(
                client,
                date_text
            )
        except Exception as error:
            print(
                f"ERA5 unavailable for {date_text}: {error}"
            )
            break

    print("ERA5 download process completed.")


if __name__ == "__main__":
    main()