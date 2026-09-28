import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)

MDAPI_DIR = os.path.join(
    PROJECT_DIR,
    "mdapi"
)

CONFIG_PATH = os.path.join(
    MDAPI_DIR,
    "config.json"
)

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "live",
    "mosdac"
)

os.makedirs(DATA_DIR, exist_ok=True)


def update_config(dataset_id, target_date):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    params = config["search_parameters"]

    params["datasetId"] = dataset_id
    params["startTime"] = target_date
    params["endTime"] = target_date
    params["boundingBox"] = ""
    params["gId"] = ""

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def run_mdapi():
    result = subprocess.run(
        [sys.executable, "mdapi.py"],
        cwd=MDAPI_DIR
    )
    return result.returncode


def main():
    target_date = (
        datetime.now() - timedelta(days=1)
    ).strftime("%Y-%m-%d")

    datasets = {
        "IMR": "3RIMG_L3G_IMR_DLY",
        "PET": "3RIMG_L3C_PET_DLY"
    }

    for name, dataset_id in datasets.items():
        print(f"Downloading MOSDAC {name}: {target_date}")

        update_config(
            dataset_id,
            target_date
        )

        code = run_mdapi()

        if code != 0:
            raise RuntimeError(
                f"MOSDAC {name} download failed with code {code}"
            )

    print("MOSDAC download completed.")


if __name__ == "__main__":
    main()