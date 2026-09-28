import os

import pandas as pd


BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        ".."
    )
)

INPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "live",
    "imd_historical"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "live",
    "imd_historical_daily"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def process_new_format(df):
    df = df.copy()

    station_col = "wigos_station_identifier"

    tmax_name = (
        "maximum_temperature_at_height_and_over_period_specified"
    )

    tmin_name = (
        "minimum_temperature_at_height_and_over_period_specified"
    )

    rainfall_name = (
        "total_precipitation_or_total_water_equivalent"
    )

    rows = []

    for station, group in df.groupby(station_col):

        latitude = group["latitude"].iloc[0]
        longitude = group["longitude"].iloc[0]

        tmax = group.loc[
            group["name"] == tmax_name,
            "value"
        ]

        tmin = group.loc[
            group["name"] == tmin_name,
            "value"
        ]

        rainfall = group.loc[
            group["name"] == rainfall_name,
            "value"
        ]

        row = {
            "station": station,
            "latitude": latitude,
            "longitude": longitude,
            "tmax": tmax.mean()
            if not tmax.empty
            else None,
            "tmin": tmin.mean()
            if not tmin.empty
            else None,
            "rainfall": rainfall.max()
            if not rainfall.empty
            else None
        }

        rows.append(row)

    return pd.DataFrame(rows)


def process_old_format(df):
    df = df.copy()

    df = df.dropna(
        subset=["station"]
    )

    grouped = []

    for station, group in df.groupby("station"):

        latitude = group["latitude"].iloc[0]
        longitude = group["longitude"].iloc[0]

        tmax = pd.to_numeric(
            group["tmax"],
            errors="coerce"
        ).dropna()

        tmin = pd.to_numeric(
            group["tmin"],
            errors="coerce"
        ).dropna()

        rainfall = pd.to_numeric(
            group["rainfall"],
            errors="coerce"
        ).dropna()

        grouped.append({
            "station": station,
            "latitude": latitude,
            "longitude": longitude,
            "tmax": tmax.mean()
            if not tmax.empty
            else None,
            "tmin": tmin.mean()
            if not tmin.empty
            else None,
            "rainfall": rainfall.max()
            if not rainfall.empty
            else None
        })

    return pd.DataFrame(grouped)


def process_file(filename):

    input_path = os.path.join(
        INPUT_DIR,
        filename
    )

    date_text = filename[4:-4]

    print(
        f"\nProcessing: {date_text}"
    )

    df = pd.read_csv(
        input_path
    )

    if "wigos_station_identifier" in df.columns:
        result = process_new_format(df)
    else:
        result = process_old_format(df)

    result["tmax"] = pd.to_numeric(
        result["tmax"],
        errors="coerce"
    )

    result["tmin"] = pd.to_numeric(
        result["tmin"],
        errors="coerce"
    )

    result["rainfall"] = pd.to_numeric(
        result["rainfall"],
        errors="coerce"
    )

    print(
        f"Stations: {len(result)}"
    )

    print(
        f"Tmax valid: {result['tmax'].notna().sum()}"
    )

    print(
        f"Tmin valid: {result['tmin'].notna().sum()}"
    )

    print(
        f"Rainfall valid: {result['rainfall'].notna().sum()}"
    )

    both = result[
        result["tmax"].notna()
        & result["tmin"].notna()
    ]

    print(
        f"Stations with both Tmax & Tmin: {len(both)}"
    )

    if not both.empty:
        print(
            f"Tmax range: "
            f"{both['tmax'].min():.1f} "
            f"to "
            f"{both['tmax'].max():.2f}"
        )

        print(
            f"Tmin range: "
            f"{both['tmin'].min():.1f} "
            f"to "
            f"{both['tmin'].max():.2f}"
        )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"imd_daily_{date_text}.csv"
    )

    result.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {output_path}"
    )


def main():

    files = sorted(
        filename
        for filename in os.listdir(INPUT_DIR)
        if (
            filename.startswith("imd_")
            and filename.endswith(".csv")
        )
    )

    for filename in files:
        process_file(filename)

    print(
        "\nIMD daily processing completed."
    )


if __name__ == "__main__":
    main()