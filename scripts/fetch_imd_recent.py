import requests
import pandas as pd
import os
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = (
    "https://wis2box.imd.gov.in/oapi/collections/"
    "urn%3Awmo%3Amd%3Ain-imd%3Asurface-based-observations.synop/items"
)

OUTPUT_DIR = r"data\live"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "imd_2026-09-20_full.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

params = {
    "f": "json",
    "bbox": "72.5,15.5,81.0,22.2",
    "datetime": "2026-09-20/2026-09-21",
    "limit": 1000,
    "offset": 0
}

all_rows = []

print("Downloading IMD observations...")
print()

while True:

    print(
        f"Requesting offset {params['offset']}..."
    )

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=60,
        verify=False
    )

    print("STATUS:", response.status_code)

    response.raise_for_status()

    data = response.json()

    features = data.get("features", [])

    print(
        "Records received:",
        len(features)
    )

    if not features:
        break

    for feature in features:

        props = feature.get(
            "properties",
            {}
        )

        coords = feature.get(
            "geometry",
            {}
        ).get(
            "coordinates",
            [None, None]
        )

        all_rows.append({

            "station":
                props.get(
                    "wigos_station_identifier"
                ),

            "name":
                props.get("name"),

            "latitude":
                coords[1],

            "longitude":
                coords[0],

            "phenomenon_time":
                props.get("phenomenonTime"),

            "report_time":
                props.get("reportTime"),

            "variable":
                props.get("name"),

            "value":
                props.get("value"),

            "units":
                props.get("units")
        })

    matched = data.get(
        "numberMatched"
    )

    if matched is not None:

        if params["offset"] + len(features) >= matched:
            break

    params["offset"] += len(features)


print()
print("==============================")
print("DOWNLOAD COMPLETED")
print("==============================")

df = pd.DataFrame(all_rows)

print("Total observations:", len(df))

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("Unique stations:")
print(df["station"].nunique())

print()
print("Variables:")
print(
    df["variable"]
    .value_counts()
    .to_string()
)