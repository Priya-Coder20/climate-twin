import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    os.path.join(
        BASE_DIR,
        "mdapi",
        "scripts",
        "live_data",
        "mosdac",
        "download_daily_mosdac.py"
    ),
    os.path.join(
        BASE_DIR,
        "scripts",
        "live_data",
        "mosdac",
        "preprocess_mosdac.py"
    ),
    os.path.join(
        BASE_DIR,
        "scripts",
        "live_data",
        "era5",
        "download_era5_live.py"
    ),
    os.path.join(
        BASE_DIR,
        "scripts",
        "live_data",
        "era5",
        "preprocess_era5.py"
    ),
    os.path.join(
        BASE_DIR,
        "scripts",
        "live_data",
        "imd",
        "download_imd_historical.py"
    ),
    os.path.join(
        BASE_DIR,
        "scripts",
        "live_data",
        "imd",
        "process_imd_historical.py"
    ),
    os.path.join(
        BASE_DIR,
        "scripts",
        "live_data",
        "imd",
        "create_imd_grids.py"
    ),
    os.path.join(
        BASE_DIR,
        "scripts",
        "live_data",
        "validate_live_inputs.py"
    ),
    os.path.join(
        BASE_DIR,
        "scripts",
        "live_data",
        "build_live_input.py"
    ),
    os.path.join(
        BASE_DIR,
        "scripts",
        "live_data",
        "run_live_prediction.py"
    )
]

def run_script(script):
    if not os.path.exists(script):
        raise FileNotFoundError(
            f"Script not found: {script}"
        )

    result = subprocess.run(
        [sys.executable, script],
        cwd=BASE_DIR
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline stopped at: {script}"
        )

def main():
    for script in SCRIPTS:
        run_script(script)

    print("DIGITAL TWIN PIPELINE COMPLETED")

if __name__ == "__main__":
    main()