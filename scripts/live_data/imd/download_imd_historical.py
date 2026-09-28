import os
from datetime import date, timedelta

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        ".."
    )
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "live",
    "imd_historical"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

BASE_URL = (
    "https://wis2box.imd.gov.in/oapi/collections/"
    "urn%3Awmo%3Amd%3Ain-imd%3Asurface-based-observations."
    "synop/items"
)


def get_existing_dates():
    dates = set()

    if not os.path.exists(OUTPUT_DIR):
        return dates

    for filename in os.listdir(OUTPUT_DIR):
        if (
            filename.startswith("imd_")
            and filename.endswith(".csv")
        ):
            date_text = filename[4:-4]

            try:
                date.fromisoformat(date_text)
                dates.add(date_text)
            except ValueError:
                continue

    return dates


def get_download_dates():
    existing = get_existing_dates()

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


def fetch_day(date_text):
    start_time = f"{date_text}T00:00:00Z"

    next_day = (
        date.fromisoformat(date_text)
        + timedelta(days=1)
    )

    end_time = (
        next_day.strftime("%Y-%m-%d")
        + "T00:00:00Z"
    )

    params = {
        "datetime": f"{start_time}/{end_time}",
        "limit": 1000
    }

    rows = []
    offset = 0

    while True:
        params["offset"] = offset

        response = requests.get(
            BASE_URL,
            params=params,
            timeout=120,
            verify=False
        )

        response.raise_for_status()

        data = response.json()

        features = data.get(
            "features",
            []
        )

        if not features:
            break

        for feature in features:
            geometry = feature.get("geometry")

            if not geometry:
                continue

            coordinates = geometry.get("coordinates")

            if not coordinates:
                continue

            properties = feature.get(
                "properties",
                {}
            )

            longitude = coordinates[0]
            latitude = coordinates[1]

            if not (
                15.63 <= latitude <= 22.02
                and
                72.65 <= longitude <= 79.84
            ):
                continue

            properties["latitude"] = latitude
            properties["longitude"] = longitude

            rows.append(properties)

        if len(features) < 1000:
            break

        offset += 1000

    return rows


def save_day(date_text, rows):
    if not rows:
        return False

    output_path = os.path.join(
        OUTPUT_DIR,
        f"imd_{date_text}.csv"
    )

    df = pd.DataFrame(rows)

    df.to_csv(
        output_path,
        index=False
    )

    return True


def main():
    dates = get_download_dates()

    if not dates:
        print("IMD data is already up to date.")
        return

    completed = 0

    for date_text in dates:
        print(
            f"Downloading IMD: {date_text}"
        )

        try:
            rows = fetch_day(date_text)

            if not save_day(
                date_text,
                rows
            ):
                print(
                    f"IMD unavailable for {date_text}."
                )
                break

            completed += 1

        except Exception as error:
            print(
                f"IMD unavailable for {date_text}: {error}"
            )
            break

    print(
        f"IMD download process completed: "
        f"{completed} day(s)"
    )


if __name__ == "__main__":
    main()