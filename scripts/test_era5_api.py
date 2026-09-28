import cdsapi

client = cdsapi.Client()

print("Requesting ERA5 wind data...")

client.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "year": "2026",
        "month": "09",
        "day": "19",
        "time": [
            "00:00",
            "06:00",
        ],
        "area": [22.1, 72.6, 15.6, 79.9],
        "data_format": "netcdf",
        "download_format": "unarchived",
    },
    "data/live/era5_wind_test_2026-09-19.zip",
)

print("ERA5 wind download completed successfully.")