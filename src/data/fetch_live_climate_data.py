import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import pandas as pd

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "live" / "api"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEZONE = ZoneInfo("Asia/Kolkata")
URL = "https://api.open-meteo.com/v1/forecast"

# Sample locations for API testing.
# This is NOT the full 65 x 73 ConvLSTM grid.
LOCATIONS = {
    "Mumbai": (19.0760, 72.8777),
    "Pune": (18.5204, 73.8567),
    "Nashik": (19.9975, 73.7898),
    "Nagpur": (21.1458, 79.0882),
    "Chhatrapati_Sambhajinagar": (19.8762, 75.3433),
    "Kolhapur": (16.7050, 74.2433),
    "Solapur": (17.6599, 75.9064),
    "Amravati": (20.9320, 77.7523),
    "Nanded": (19.1383, 77.3210),
    "Ratnagiri": (16.9902, 73.3120),
}


def fetch_location(name, latitude, longitude, today):
    """Fetch recent daily data and a short forecast."""

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
        ],
        "past_days": 7,
        "forecast_days": 3,
        "timezone": "Asia/Kolkata",
    }

    response = requests.get(URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    daily = payload.get("daily", {})
    dates = daily.get("time", [])

    required = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
    ]
    for variable in required:
        if variable not in daily:
            raise RuntimeError(
                f"Missing {variable} in API response for {name}"
            )
        if len(daily[variable]) != len(dates):
            raise RuntimeError(
                f"Inconsistent date/value count for {name}: {variable}"
            )

    if not dates:
        raise RuntimeError(f"No daily data returned for {name}")

    retrieved_at = datetime.now(TIMEZONE).isoformat()
    rows = []

    for i, day_string in enumerate(dates):
        day = datetime.strptime(day_string, "%Y-%m-%d").date()

        # Do not silently label forecast data as observations.
        # Past dates are API historical/recent estimates, not necessarily
        # station-observed measurements.
        if day < today:
            data_type = "past_estimate"
        elif day == today:
            data_type = "current_day_estimate"
        else:
            data_type = "forecast"

        tmax = daily["temperature_2m_max"][i]
        tmin = daily["temperature_2m_min"][i]
        rainfall = daily["precipitation_sum"][i]

        # Reject impossible/incomplete temperature pairs.
        if (
            tmax is not None
            and tmin is not None
            and float(tmax) < float(tmin)
        ):
            raise RuntimeError(
                f"Maximum temperature below minimum for {name} on {day}"
            )

        rows.append({
            "date": day_string,
            "location": name,
            "latitude": latitude,
            "longitude": longitude,
            "tmax_c": tmax,
            "tmin_c": tmin,
            "rainfall_mm": rainfall,
            "data_type": data_type,
            "source": "Open-Meteo",
            "retrieved_at": retrieved_at,
        })

    return rows, payload


def main():
    today = datetime.now(TIMEZONE).date()
    all_rows = []
    raw_responses = {}
    failures = []

    print(f"India date: {today}")
    print("Source: Open-Meteo")
    print("Fetching daily temperature and precipitation...\n")

    for name, (latitude, longitude) in LOCATIONS.items():
        print(f"Fetching {name}...")

        try:
            rows, payload = fetch_location(
                name, latitude, longitude, today
            )
            all_rows.extend(rows)
            raw_responses[name] = payload
            print(f"  OK: received {len(rows)} daily records")

        except (requests.RequestException, ValueError, RuntimeError) as exc:
            failures.append({"location": name, "error": str(exc)})
            print(f"  FAILED: {exc}")

    if not all_rows:
        raise RuntimeError(
            "No data fetched. Check your internet connection and API."
        )

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(["date", "location"])

    # Save all returned records, with explicit data-type labels.
    all_path = OUT_DIR / "latest_climate_data.csv"
    df.to_csv(all_path, index=False)

    # Save only dates up to today; these are NOT guaranteed to be
    # official observed measurements.
    non_future = df[df["date"] <= today].copy()
    non_future_path = OUT_DIR / "latest_nonfuture_data.csv"
    non_future.to_csv(non_future_path, index=False)

    # Keep future forecast dates in a separate file.
    forecast = df[df["date"] > today].copy()
    forecast_path = OUT_DIR / "forecast_data.csv"
    forecast.to_csv(forecast_path, index=False)

    metadata = {
        "retrieved_at": datetime.now(TIMEZONE).isoformat(),
        "india_today": today.isoformat(),
        "source": "Open-Meteo Forecast API",
        "locations_requested": len(LOCATIONS),
        "locations_succeeded": len(raw_responses),
        "locations_failed": failures,
        "record_count": len(df),
        "latest_returned_date": str(df["date"].max()),
        "note": (
            "Past dates are API estimates, not guaranteed station "
            "observations. Current-day values may be incomplete. "
            "Future dates are forecasts. These point samples are not "
            "the 65 x 73 model grid."
        ),
    }

    (OUT_DIR / "raw_api_responses.json").write_text(
        json.dumps(raw_responses, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "pipeline_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"India today: {today}")
    print(f"Total records: {len(df)}")
    print(f"Locations successful: {len(raw_responses)}/{len(LOCATIONS)}")
    print(f"Latest returned date: {df['date'].max()}")
    print(f"Future forecast records: {len(forecast)}")
    print(f"Locations failed: {len(failures)}")
    print(f"\nAll data: {all_path}")
    print(f"Non-future data: {non_future_path}")
    print(f"Forecast only: {forecast_path}")
    print(f"Metadata: {OUT_DIR / 'pipeline_metadata.json'}")

    print("\nLatest available date by location:")
    print(
        df.groupby("location")["date"]
        .max()
        .to_string()
    )

    if failures:
        print("\nCheck failed locations in pipeline_metadata.json")


if __name__ == "__main__":
    main()