"""
AI Digital Twin of Maharashtra Climate - Backend
================================================
FastAPI backend service for inspecting live climate datasets, verifying model inputs,
and serving spatial climate dashboard data.

Author: AI Climate Twin Engineering Team
"""

import os
import sys
import re
import csv
import json
import logging
import numpy as np
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import xarray as xr
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("climate-twin-backend")

# ==============================================================================
# PROJECT ROOT RESOLUTION (Relative Path Safety)
# ==============================================================================

def resolve_project_root() -> Path:
    """
    Dynamically discover the repository root directory.
    Checks:
      1. CLIMATE_TWIN_ROOT environment variable (if explicitly set).
      2. Current working directory (if it contains 'data' and 'models').
      3. Parent directory of this file (backend/..).
    """
    env_root = os.getenv("CLIMATE_TWIN_ROOT")
    if env_root and Path(env_root).is_dir():
        logger.info(f"Resolved project root from CLIMATE_TWIN_ROOT: {env_root}")
        return Path(env_root).resolve()

    cwd = Path.cwd().resolve()
    if (cwd / "data").is_dir() and (cwd / "models").is_dir():
        logger.info(f"Resolved project root from CWD: {cwd}")
        return cwd

    file_root = Path(__file__).resolve().parent.parent
    if (file_root / "data").is_dir():
        logger.info(f"Resolved project root from file hierarchy: {file_root}")
        return file_root

    logger.warning(f"Could not verify project root signatures. Defaulting to CWD: {cwd}")
    return cwd

PROJECT_ROOT = resolve_project_root()
DATA_LIVE_DIR = PROJECT_ROOT / "data" / "live"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SPATIAL_MAPS_DIR = OUTPUTS_DIR / "spatial_maps"
EXPLAINABILITY_DIR = OUTPUTS_DIR / "explainability"
LIVE_PREDICTION_DIR = OUTPUTS_DIR / "live_prediction"

# Expected 10 model channels across sources
EXPECTED_CHANNELS = [
    "imr",
    "pet",
    "era5_temperature",
    "era5_dewpoint",
    "era5_wind_speed",
    "era5_pressure",
    "era5_rainfall",
    "imd_max_temperature",
    "imd_min_temperature",
    "imd_rainfall",
]

REQUIRED_INPUT_DAYS = 7

# ==============================================================================
# FASTAPI APP & CORS
# ==============================================================================

app = FastAPI(
    title="AI Digital Twin of Maharashtra Climate API",
    description="Backend service providing live data inspection, spatial predictions, and explainability.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration for local Vite frontend and common development ports
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# HELPER FUNCTIONS: DYNAMIC DATE & CHANNEL DISCOVERY
# ==============================================================================

MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}

def parse_mosdac_raw_date(filename: str) -> Optional[str]:
    """
    Parses dates from raw MOSDAC filenames, e.g.:
    3RIMG_14SEP2026_0015_L3C_PET_DLY_V01R00.h5 -> 2026-09-14
    """
    match = re.search(r"3RIMG_(\d{2})([A-Z]{3})(\d{4})_", filename)
    if match:
        day, mon, year = match.groups()
        mon_num = MONTH_MAP.get(mon.upper())
        if mon_num:
            return f"{year}-{mon_num}-{day}"
    return None

def get_file_metadata(filepath: Path) -> Dict[str, Any]:
    """Safe metadata extraction (size and modification time) for any file."""
    try:
        stat = filepath.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        return {
            "path": str(filepath.relative_to(PROJECT_ROOT)),
            "size_bytes": stat.st_size,
            "modified_time_utc": mtime,
            "is_valid": stat.st_size > 0,
        }
    except Exception as e:
        return {
            "path": str(filepath),
            "error": str(e),
            "is_valid": False,
        }

def inspect_mosdac_data(live_dir: Path) -> Dict[str, Any]:
    """Dynamically inspect raw and processed MOSDAC files and channels."""
    mosdac_dir = live_dir / "mosdac"
    processed_dir = mosdac_dir / "processed"

    result: Dict[str, Any] = {
        "source": "MOSDAC",
        "folder_exists": mosdac_dir.is_dir(),
        "processed_folder_exists": processed_dir.is_dir(),
        "latest_raw_date": None,
        "latest_processed_date": None,
        "latest_date": None,
        "raw_dates": [],
        "processed_dates": [],
        "channels_by_date": {},
        "raw_files_count": 0,
        "processed_files_count": 0,
        "latest_file_modified_utc": None,
        "issues": [],
    }

    if not mosdac_dir.is_dir():
        result["issues"].append(f"Directory missing: {mosdac_dir}")
        return result

    latest_mtime: float = 0.0

    # Scan raw MOSDAC HDF5 files
    raw_dates: Set[str] = set()
    try:
        for item in mosdac_dir.iterdir():
            if item.is_file() and item.suffix.lower() == ".h5" and item.stat().st_size > 0:
                result["raw_files_count"] += 1
                latest_mtime = max(latest_mtime, item.stat().st_mtime)
                d = parse_mosdac_raw_date(item.name)
                if d:
                    raw_dates.add(d)
    except Exception as e:
        result["issues"].append(f"Error scanning raw MOSDAC files: {str(e)}")

    # Scan processed MOSDAC .npy files
    processed_dates: Set[str] = set()
    channels_by_date: Dict[str, Set[str]] = {}

    if processed_dir.is_dir():
        try:
            for item in processed_dir.iterdir():
                if item.is_file() and item.suffix.lower() == ".npy" and item.stat().st_size > 0:
                    result["processed_files_count"] += 1
                    latest_mtime = max(latest_mtime, item.stat().st_mtime)
                    match = re.match(r"^(imr|pet)_(\d{4}-\d{2}-\d{2})\.npy$", item.name)
                    if match:
                        channel, d = match.groups()
                        processed_dates.add(d)
                        if d not in channels_by_date:
                            channels_by_date[d] = set()
                        channels_by_date[d].add(channel)
        except Exception as e:
            result["issues"].append(f"Error scanning processed MOSDAC files: {str(e)}")

    sorted_raw = sorted(raw_dates)
    sorted_proc = sorted(processed_dates)

    result["raw_dates"] = sorted_raw
    result["processed_dates"] = sorted_proc
    result["latest_raw_date"] = sorted_raw[-1] if sorted_raw else None
    result["latest_processed_date"] = sorted_proc[-1] if sorted_proc else None

    # Overall latest available date for MOSDAC
    all_dates = sorted(raw_dates.union(processed_dates))
    result["latest_date"] = all_dates[-1] if all_dates else None
    result["channels_by_date"] = {k: sorted(list(v)) for k, v in channels_by_date.items()}

    if latest_mtime > 0:
        result["latest_file_modified_utc"] = datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat()

    return result

def inspect_era5_data(live_dir: Path) -> Dict[str, Any]:
    """Dynamically inspect raw date folders and processed ERA5 files."""
    era5_dir = live_dir / "era5"
    processed_dir = era5_dir / "processed"

    result: Dict[str, Any] = {
        "source": "ERA5",
        "folder_exists": era5_dir.is_dir(),
        "processed_folder_exists": processed_dir.is_dir(),
        "latest_raw_date": None,
        "latest_processed_date": None,
        "latest_date": None,
        "raw_dates": [],
        "processed_dates": [],
        "channels_by_date": {},
        "empty_raw_folders": [],
        "raw_files_count": 0,
        "processed_files_count": 0,
        "latest_file_modified_utc": None,
        "issues": [],
    }

    if not era5_dir.is_dir():
        result["issues"].append(f"Directory missing: {era5_dir}")
        return result

    latest_mtime: float = 0.0

    # Scan raw ERA5 subdirectories (named YYYY-MM-DD)
    raw_dates: Set[str] = set()
    try:
        for sub in era5_dir.iterdir():
            if sub.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", sub.name):
                # Verify subfolder contains non-empty data files
                non_empty_files = [f for f in sub.iterdir() if f.is_file() and f.stat().st_size > 0]
                if non_empty_files:
                    raw_dates.add(sub.name)
                    result["raw_files_count"] += len(non_empty_files)
                    for f in non_empty_files:
                        latest_mtime = max(latest_mtime, f.stat().st_mtime)
                else:
                    result["empty_raw_folders"].append(sub.name)
    except Exception as e:
        result["issues"].append(f"Error scanning raw ERA5 directories: {str(e)}")

    # Scan processed ERA5 .npy files
    processed_dates: Set[str] = set()
    channels_by_date: Dict[str, Set[str]] = {}

    if processed_dir.is_dir():
        try:
            for item in processed_dir.iterdir():
                if item.is_file() and item.suffix.lower() == ".npy" and item.stat().st_size > 0:
                    result["processed_files_count"] += 1
                    latest_mtime = max(latest_mtime, item.stat().st_mtime)
                    match = re.match(r"^era5_(temperature|dewpoint|wind_speed|pressure|rainfall)_(\d{4}-\d{2}-\d{2})\.npy$", item.name)
                    if match:
                        var_name, d = match.groups()
                        processed_dates.add(d)
                        channel_key = f"era5_{var_name}"
                        if d not in channels_by_date:
                            channels_by_date[d] = set()
                        channels_by_date[d].add(channel_key)
        except Exception as e:
            result["issues"].append(f"Error scanning processed ERA5 files: {str(e)}")

    sorted_raw = sorted(raw_dates)
    sorted_proc = sorted(processed_dates)

    result["raw_dates"] = sorted_raw
    result["processed_dates"] = sorted_proc
    result["latest_raw_date"] = sorted_raw[-1] if sorted_raw else None
    result["latest_processed_date"] = sorted_proc[-1] if sorted_proc else None

    # Overall latest available date for ERA5
    all_dates = sorted(raw_dates.union(processed_dates))
    result["latest_date"] = all_dates[-1] if all_dates else None
    result["channels_by_date"] = {k: sorted(list(v)) for k, v in channels_by_date.items()}

    if latest_mtime > 0:
        result["latest_file_modified_utc"] = datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat()

    return result

def inspect_imd_data(live_dir: Path) -> Dict[str, Any]:
    """Dynamically inspect raw CSV files and processed IMD files."""
    historical_dir = live_dir / "imd_historical"
    processed_dir = live_dir / "imd_processed"

    result: Dict[str, Any] = {
        "source": "IMD",
        "historical_folder_exists": historical_dir.is_dir(),
        "processed_folder_exists": processed_dir.is_dir(),
        "latest_raw_date": None,
        "latest_processed_date": None,
        "latest_date": None,
        "raw_dates": [],
        "processed_dates": [],
        "channels_by_date": {},
        "raw_files_count": 0,
        "processed_files_count": 0,
        "latest_file_modified_utc": None,
        "issues": [],
    }

    latest_mtime: float = 0.0

    # Scan raw IMD CSV files
    raw_dates: Set[str] = set()
    if historical_dir.is_dir():
        try:
            for item in historical_dir.iterdir():
                if item.is_file() and item.suffix.lower() == ".csv" and item.stat().st_size > 0:
                    result["raw_files_count"] += 1
                    latest_mtime = max(latest_mtime, item.stat().st_mtime)
                    match = re.match(r"^imd_(\d{4}-\d{2}-\d{2})\.csv$", item.name)
                    if match:
                        raw_dates.add(match.group(1))
        except Exception as e:
            result["issues"].append(f"Error scanning raw IMD CSVs: {str(e)}")
    else:
        result["issues"].append(f"Historical IMD directory missing: {historical_dir}")

    # Scan processed IMD .npy files
    processed_dates: Set[str] = set()
    channels_by_date: Dict[str, Set[str]] = {}

    imd_var_map = {
        "tmax": "imd_max_temperature",
        "tmin": "imd_min_temperature",
        "rainfall": "imd_rainfall",
    }

    if processed_dir.is_dir():
        try:
            for item in processed_dir.iterdir():
                if item.is_file() and item.suffix.lower() == ".npy" and item.stat().st_size > 0:
                    result["processed_files_count"] += 1
                    latest_mtime = max(latest_mtime, item.stat().st_mtime)
                    match = re.match(r"^(tmax|tmin|rainfall)_(\d{4}-\d{2}-\d{2})\.npy$", item.name)
                    if match:
                        var_key, d = match.groups()
                        processed_dates.add(d)
                        channel_name = imd_var_map.get(var_key, var_key)
                        if d not in channels_by_date:
                            channels_by_date[d] = set()
                        channels_by_date[d].add(channel_name)
        except Exception as e:
            result["issues"].append(f"Error scanning processed IMD files: {str(e)}")
    else:
        result["issues"].append(f"Processed IMD directory missing: {processed_dir}")

    sorted_raw = sorted(raw_dates)
    sorted_proc = sorted(processed_dates)

    result["raw_dates"] = sorted_raw
    result["processed_dates"] = sorted_proc
    result["latest_raw_date"] = sorted_raw[-1] if sorted_raw else None
    result["latest_processed_date"] = sorted_proc[-1] if sorted_proc else None

    # Overall latest available date for IMD
    all_dates = sorted(raw_dates.union(processed_dates))
    result["latest_date"] = all_dates[-1] if all_dates else None
    result["channels_by_date"] = {k: sorted(list(v)) for k, v in channels_by_date.items()}

    if latest_mtime > 0:
        result["latest_file_modified_utc"] = datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat()

    return result

def inspect_system_status(project_root: Path) -> Dict[str, Any]:
    """
    Comprehensive dynamic inspection of live-data folders, channel alignment,
    and forecast feasibility.
    """
    live_dir = project_root / "data" / "live"

    if not live_dir.is_dir():
        return {
            "error": "Live data directory not found",
            "live_data_dir": str(live_dir),
            "status": "missing_directory",
            "is_sufficient_for_forecast": False,
        }

    # Inspect each data provider
    mosdac_info = inspect_mosdac_data(live_dir)
    era5_info = inspect_era5_data(live_dir)
    imd_info = inspect_imd_data(live_dir)

    # Aggregate channels per date across all 3 sources
    all_dates: Set[str] = set()
    all_dates.update(mosdac_info.get("channels_by_date", {}).keys())
    all_dates.update(era5_info.get("channels_by_date", {}).keys())
    all_dates.update(imd_info.get("channels_by_date", {}).keys())

    expected_set = set(EXPECTED_CHANNELS)
    complete_dates: List[str] = []
    partial_dates: Dict[str, Dict[str, Any]] = {}

    for d in sorted(all_dates):
        channels_present: Set[str] = set()
        channels_present.update(mosdac_info.get("channels_by_date", {}).get(d, []))
        channels_present.update(era5_info.get("channels_by_date", {}).get(d, []))
        channels_present.update(imd_info.get("channels_by_date", {}).get(d, []))

        missing = sorted(list(expected_set - channels_present))
        present = sorted(list(channels_present))

        if len(missing) == 0:
            complete_dates.append(d)
        else:
            partial_dates[d] = {
                "available_channels": present,
                "missing_channels": missing,
                "available_count": len(present),
                "total_required": len(EXPECTED_CHANNELS),
            }

    latest_common_processed_date = complete_dates[-1] if complete_dates else None

    # Calculate latest common raw date across all 3 sources
    raw_dates_mosdac = set(mosdac_info.get("raw_dates", []))
    raw_dates_era5 = set(era5_info.get("raw_dates", []))
    raw_dates_imd = set(imd_info.get("raw_dates", []))
    common_raw_dates = sorted(raw_dates_mosdac.intersection(raw_dates_era5).intersection(raw_dates_imd))
    latest_common_raw_date = common_raw_dates[-1] if common_raw_dates else None

    # Consecutive daily sequence check (model requires 7 days)
    consecutive_sequence: List[str] = []
    if complete_dates:
        # Trace backwards from latest common date to check consecutive days
        sorted_complete = sorted(complete_dates)
        consecutive_sequence = [sorted_complete[-1]]
        cur = datetime.strptime(sorted_complete[-1], "%Y-%m-%d")
        for prev_date_str in reversed(sorted_complete[:-1]):
            prev_dt = datetime.strptime(prev_date_str, "%Y-%m-%d")
            if (cur - prev_dt).days == 1:
                consecutive_sequence.insert(0, prev_date_str)
                cur = prev_dt
            else:
                break

    is_7day_window_available = len(consecutive_sequence) >= REQUIRED_INPUT_DAYS

    # Determine freshness and feasibility
    # New raw data exists if ERA5 raw > latest_common or IMD raw > latest_common
    new_data_sources: List[str] = []
    if era5_info.get("latest_raw_date") and latest_common_processed_date:
        if era5_info["latest_raw_date"] > latest_common_processed_date:
            new_data_sources.append(
                f"ERA5 raw files available through {era5_info['latest_raw_date']} ({era5_info['latest_raw_date']} > {latest_common_processed_date})"
            )
    if imd_info.get("latest_raw_date") and latest_common_processed_date:
        if imd_info["latest_raw_date"] > latest_common_processed_date:
            new_data_sources.append(
                f"IMD raw files available through {imd_info['latest_raw_date']} ({imd_info['latest_raw_date']} > {latest_common_processed_date})"
            )
    if imd_info.get("latest_processed_date") and latest_common_processed_date:
        if imd_info["latest_processed_date"] > latest_common_processed_date:
            new_data_sources.append(
                f"IMD processed data available through {imd_info['latest_processed_date']} (partially preprocessed)"
            )

    has_new_unaligned_data = len(new_data_sources) > 0

    # Is data stale or insufficient?
    # MOSDAC terminates on 2026-09-20; no MOSDAC data exists past this date.
    mosdac_latest = mosdac_info.get("latest_date")
    is_stale = False
    is_sufficient = False
    status_label = "ready"
    reason = ""

    if not latest_common_processed_date:
        is_stale = True
        is_sufficient = False
        status_label = "insufficient"
        reason = "No common date exists where all 10 required climate channels are available."
    elif not is_7day_window_available:
        is_stale = True
        is_sufficient = False
        status_label = "insufficient"
        reason = f"Only {len(consecutive_sequence)} consecutive complete days available; ConvLSTM model requires {REQUIRED_INPUT_DAYS} days."
    else:
        # A 7-day aligned historical block exists ending at latest_common_processed_date.
        # But can a NEW forecast be generated for future dates?
        if has_new_unaligned_data:
            is_stale = True
            is_sufficient = False
            status_label = "stale_inputs"
            reason = (
                f"New observation data has arrived for ERA5 (up to {era5_info.get('latest_raw_date')}) and "
                f"IMD (up to {imd_info.get('latest_raw_date')}), but MOSDAC satellite observations terminate on "
                f"{mosdac_latest}. Because all 10 channels are mandatory for the 7-day model window, inputs for "
                f"any forecast beyond {latest_common_processed_date} are incomplete and insufficient."
            )
        else:
            is_stale = False
            is_sufficient = True
            status_label = "aligned"
            reason = f"All 10 channels are fully synchronized through {latest_common_processed_date}."

    return {
        "status": status_label,
        "latest_available_dates": {
            "mosdac": mosdac_info.get("latest_date"),
            "era5": era5_info.get("latest_date"),
            "imd": imd_info.get("latest_date"),
        },
        "latest_common_date": {
            "processed_channels_all_10": latest_common_processed_date,
            "raw_sources_all_3": latest_common_raw_date,
        },
        "expected_channels": EXPECTED_CHANNELS,
        "consecutive_input_window": {
            "dates": consecutive_sequence,
            "length_days": len(consecutive_sequence),
            "required_days": REQUIRED_INPUT_DAYS,
            "is_window_complete": is_7day_window_available,
        },
        "forecast_feasibility": {
            "new_data_available": has_new_unaligned_data,
            "new_data_details": new_data_sources,
            "is_stale": is_stale,
            "is_sufficient_for_forecast": is_sufficient,
            "reason": reason,
        },
        "channel_gap_analysis_for_unaligned_dates": partial_dates,
        "sources_detail": {
            "mosdac": mosdac_info,
            "era5": era5_info,
            "imd": imd_info,
        },
    }

# ==============================================================================
# API ROUTES
# ==============================================================================

@app.get("/", summary="API Root")
def get_root() -> Dict[str, Any]:
    """Welcome and overview of the Maharashtra Climate Digital Twin API."""
    return {
        "service": "AI Digital Twin of Maharashtra Climate",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
        "endpoints": {
            "health": "/api/health",
            "status": "/api/status",
            "prediction": "/api/prediction",
            "explainability_features": "/api/explainability/features",
            "spatial_maps": "/api/maps/{map_type}",
        },
    }

@app.get("/api/health", summary="Health Check")
def get_health() -> Dict[str, Any]:
    """
    Verifies that the backend server is operational and can access project directories.
    """
    try:
        live_exists = DATA_LIVE_DIR.is_dir()
        outputs_exists = OUTPUTS_DIR.is_dir()
        return {
            "status": "healthy",
            "service": "AI Digital Twin of Maharashtra Climate Backend",
            "version": "1.0.0",
            "project_root": str(PROJECT_ROOT),
            "data_live_directory": {
                "path": str(DATA_LIVE_DIR.relative_to(PROJECT_ROOT)),
                "exists": live_exists,
            },
            "outputs_directory": {
                "path": str(OUTPUTS_DIR.relative_to(PROJECT_ROOT)),
                "exists": outputs_exists,
            },
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Health check encountered error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@app.get("/api/status", summary="Live Data & Pipeline Status")
def get_status() -> Dict[str, Any]:
    """
    Inspects actual live-data folders and files dynamically:
    - Latest available dates for MOSDAC, ERA5, and IMD.
    - Latest common date across all required 10 climate features.
    - Which expected channels are present or missing per date.
    - Accurate assessment of whether inputs are fresh, stale, or insufficient.
    """
    try:
        inspection_result = inspect_system_status(PROJECT_ROOT)
        return inspection_result
    except Exception as e:
        logger.error(f"Error inspecting system status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to inspect live data status: {str(e)}"
        )

@app.get("/api/prediction", summary="Temperature Prediction Endpoint")
def get_prediction(
    include_historical: bool = Query(
        default=False,
        description="If True, includes the verified historical evaluation artifact from outputs/live_prediction if available on disk."
    )
) -> Dict[str, Any]:
    """
    Prediction endpoint with strict non-fabrication guarantee:
    - Does NOT fabricate unverified future forecasts.
    - Returns a clear availability status explaining input alignment and feasibility.
    - Optionally serves verified offline evaluation artifacts from disk without misrepresenting them as live forecasts.
    - Maintains compatibility with React/Vite frontend response structure (prediction_date, tmax, tmin).
    """
    try:
        # Run actual dynamic status check
        status_info = inspect_system_status(PROJECT_ROOT)
        feasibility = status_info.get("forecast_feasibility", {})
        common_date = status_info.get("latest_common_date", {}).get("processed_channels_all_10")

        # Check if an existing verified evaluation artifact exists on disk
        metadata_file = LIVE_PREDICTION_DIR / "prediction_metadata.json"
        artifact_data: Optional[Dict[str, Any]] = None

        if metadata_file.is_file():
            try:
                with open(metadata_file, "r") as f:
                    artifact_data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read prediction_metadata.json: {e}")

        # Construct clear availability response
        # Live forecast cannot be computed without fabricated data because MOSDAC terminates on 2026-09-20.
        response: Dict[str, Any] = {
            "status": "unavailable",
            "available": False,
            "message": (
                "Live future forecast is unavailable. While historical input data is aligned through "
                f"{common_date or 'an earlier date'}, subsequent satellite observations (MOSDAC IMR/PET) "
                "are missing. To avoid fabricated climate projections, real-time forecast generation is withheld "
                "until future multi-source inputs are synchronized."
            ),
            "prediction_date": None,
            "tmax": None,
            "tmin": None,
            "alignment_summary": {
                "latest_common_input_date": common_date,
                "is_stale": feasibility.get("is_stale", True),
                "is_sufficient_for_forecast": False,
                "reason": feasibility.get("reason"),
            },
            "verified_disk_artifact": None,
        }

        # If a verified disk artifact exists, report its details transparently
        if artifact_data:
            artifact_date = artifact_data.get("prediction_date")
            artifact_tmax = {
                "min": artifact_data.get("tmax_min"),
                "max": artifact_data.get("tmax_max"),
                "mean": artifact_data.get("tmax_mean"),
            }
            artifact_tmin = {
                "min": artifact_data.get("tmin_min"),
                "max": artifact_data.get("tmin_max"),
                "mean": artifact_data.get("tmin_mean"),
            }

            response["verified_disk_artifact"] = {
                "available": True,
                "prediction_date": artifact_date,
                "model": artifact_data.get("model"),
                "input_dates": artifact_data.get("input_dates"),
                "tmax": artifact_tmax,
                "tmin": artifact_tmin,
                "type": "offline_evaluation_artifact",
                "notice": "This reflects a previously evaluated run on disk, not a freshly generated live forecast.",
            }

            # If the client explicitly requests historical evaluation mode, populate top-level fields
            if include_historical:
                response["status"] = "historical_artifact"
                response["available"] = True
                response["prediction_date"] = artifact_date
                response["tmax"] = artifact_tmax
                response["tmin"] = artifact_tmin
                response["message"] = (
                    f"Serving verified offline evaluation artifact for {artifact_date} "
                    f"generated from inputs {artifact_data.get('input_dates', [])}."
                )

        return response

    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction handler failed: {str(e)}"
        )

@app.get("/api/explainability/features", summary="Feature Importance Explainability")
def get_explainability_features() -> Dict[str, Any]:
    """
    Returns feature importance percentages from outputs/explainability/feature_importance.csv.
    Maintains compatibility with the frontend's expected featureRows mapping:
    {
      "tmax": { "FeatureName": value, ... },
      "tmin": { "FeatureName": value, ... }
    }
    """
    csv_path = EXPLAINABILITY_DIR / "feature_importance.csv"

    if not csv_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature importance file not found at {csv_path.relative_to(PROJECT_ROOT) if PROJECT_ROOT in csv_path.parents else csv_path}"
        )

    tmax_factors: Dict[str, float] = {}
    tmin_factors: Dict[str, float] = {}

    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feat = row.get("Feature", "").strip()
                if not feat:
                    continue
                try:
                    tmax_val = float(row.get("Tmax Importance (%)", 0.0))
                    tmin_val = float(row.get("Tmin Importance (%)", 0.0))
                    tmax_factors[feat] = round(tmax_val, 3)
                    tmin_factors[feat] = round(tmin_val, 3)
                except ValueError:
                    continue

        return {
            "status": "success",
            "source_file": str(csv_path.relative_to(PROJECT_ROOT)),
            "tmax": tmax_factors,
            "tmin": tmin_factors,
        }
    except Exception as e:
        logger.error(f"Error reading feature importance: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse feature importance: {str(e)}"
        )

@app.get("/api/maps/{map_type}", summary="Spatial Temperature Map Overlay")
def get_spatial_map(map_type: str):
    """
    Serves spatial prediction map PNGs (tmax or tmin) for the Leaflet image overlay.
    """
    normalized_type = map_type.lower().strip()
    if normalized_type not in ["tmax", "tmin"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid map type '{map_type}'. Allowed types are 'tmax' and 'tmin'."
        )

    map_path = SPATIAL_MAPS_DIR / f"{normalized_type}_map.png"

    if not map_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Spatial map image '{normalized_type}_map.png' not found."
        )

    return FileResponse(
        path=str(map_path),
        media_type="image/png",
        filename=f"{normalized_type}_map.png"
    )

# ==============================================================================
# MAIN ENTRYPOINT (For Direct Execution)
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server from project root: {PROJECT_ROOT}")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)


@app.get("/api/historical-dates")
def get_historical_dates():
    dataset_path = PROJECT_ROOT / "data" / "processed" / "master_climate_dataset.nc"

    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Historical dataset not found.")

    try:
        with xr.open_dataset(dataset_path) as ds:
            if "time" not in ds.coords:
                raise HTTPException(status_code=500, detail="Dataset has no time coordinate.")

            dates = np.datetime_as_string(
                ds["time"].values.astype("datetime64[D]"), unit="D"
            ).tolist()

        return {
            "dates": dates,
            "start_date": dates[0] if dates else None,
            "end_date": dates[-1] if dates else None,
            "source": "master_climate_dataset.nc",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read historical dates: {exc}") from exc

@app.get("/api/live-weather")
def get_live_weather(
    latitude: float = 21.1458,
    longitude: float = 79.0882,
):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,relative_humidity_2m,"
            "apparent_temperature,precipitation,"
            "rain,weather_code,wind_speed_10m"
        ),
        "daily": (
            "temperature_2m_max,temperature_2m_min,"
            "precipitation_sum"
        ),
        "timezone": "Asia/Kolkata",
        "forecast_days": 7,
    }

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        + urlencode(params)
    )

    try:
        with urlopen(url, timeout=15) as response:
            weather_data = json.loads(
                response.read().decode("utf-8")
            )

        return {
            "source": "Open-Meteo",
            "location": "Nagpur, Maharashtra",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "data": weather_data,
        }

    except HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Weather provider returned HTTP {exc.code}",
        ) from exc

    except (URLError, TimeoutError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the weather provider.",
        ) from exc

    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Weather provider returned invalid JSON.",
        ) from exc


@app.get("/api/historical-grid")
def get_historical_grid(variable: str = "tmax", date: str = Query(...)):
    variable_names = {
        "tmax": "imd_max_temperature",
        "max_temperature": "imd_max_temperature",
        "tmin": "imd_min_temperature",
        "min_temperature": "imd_min_temperature",
        "rainfall": "imd_rainfall",
    }

    if variable not in variable_names:
        raise HTTPException(status_code=400, detail="Unsupported historical climate variable.")

    dataset_path = PROJECT_ROOT / "data" / "processed" / "master_climate_dataset.nc"

    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Historical dataset not found.")

    try:
        requested_date = np.datetime64(date, "D")

        with xr.open_dataset(dataset_path) as ds:
            if variable_names[variable] not in ds.data_vars:
                raise HTTPException(status_code=404, detail="Climate variable not found.")

            available = ds["time"].values.astype("datetime64[D]")
            matches = np.flatnonzero(available == requested_date)

            if len(matches) == 0:
                raise HTTPException(status_code=404, detail=f"No historical data for {date}.")

            frame = ds[variable_names[variable]].isel(time=int(matches[0]))
            values = np.asarray(frame.values, dtype=np.float64)

            latitudes = np.asarray(ds["latitude"].values, dtype=float)
            longitudes = np.asarray(ds["longitude"].values, dtype=float)

            if "maharashtra_mask" in ds:
                mask = np.asarray(ds["maharashtra_mask"].values, dtype=bool)
                values[~mask] = np.nan

        values[~np.isfinite(values)] = np.nan

        return {
            "variable": variable,
            "prediction_date": date,
            "units": "mm" if variable == "rainfall" else "°C",
            "source": f"master dataset variable: {variable_names[variable]}",
            "latitudes": latitudes.tolist(),
            "longitudes": longitudes.tolist(),
            "values": [
                [float(v) if np.isfinite(v) else None for v in row]
                for row in values
            ],
        }
    except HTTPException:
        raise
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date: {date}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load historical grid: {exc}") from exc


@app.get("/api/prediction-dates")
def get_prediction_dates():
    project_root = Path(__file__).resolve().parent.parent

    dates_path = (
        project_root
        / "data"
        / "processed_daily"
        / "cnn_dataset"
        / "multivariable_prepared"
        / "target_dates_test.npy"
    )

    live_dir = project_root / "outputs" / "live_prediction"
    forecast_file = live_dir / "live_forecast.npy"
    metadata_path = live_dir / "live_forecast_metadata.json"

    dates = set()

    # Preserve existing historical prediction dates.
    if dates_path.exists():
        saved_dates = np.load(dates_path, allow_pickle=True)
        dates.update(
            str(d.decode("utf-8") if isinstance(d, bytes) else d)[:10]
            for d in saved_dates
        )

    # Add dates from the new AI forecast.
    if forecast_file.exists() and metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            if (
                metadata.get("status") == "generated"
                and metadata.get("is_current_forecast") is True
            ):
                dates.update(
                    str(d)[:10]
                    for d in metadata.get("forecast_dates", [])
                )
        except (OSError, ValueError):
            pass

    if not dates:
        raise HTTPException(
            status_code=404,
            detail="No historical or generated prediction dates found."
        )

    return {"dates": sorted(dates)}


@app.get("/api/prediction-grid")
def get_prediction_grid(variable: str = "tmax", date: str = ""):
    aliases = {
        "max_temperature": "tmax",
        "min_temperature": "tmin",
    }
    variable = aliases.get(variable, variable)

    variables = {"tmax": 0, "tmin": 1, "rainfall": 2}

    if variable not in variables:
        raise HTTPException(
            status_code=400,
            detail="variable must be tmax, tmin, or rainfall"
        )

    project_root = Path(__file__).resolve().parent.parent
    spatial_dir = project_root / "outputs" / "spatial_maps"
    live_dir = project_root / "outputs" / "live_prediction"

    evaluation_dir = (
        project_root / "outputs" / "multivariable_convlstm_evaluation"
    )
    prepared_dir = (
        project_root / "data" / "processed_daily" / "cnn_dataset"
        / "multivariable_prepared"
    )

    try:
        latitudes = np.asarray(
            np.load(spatial_dir / "latitude.npy")
        ).squeeze()
        longitudes = np.asarray(
            np.load(spatial_dir / "longitude.npy")
        ).squeeze()

        values = None
        prediction_date = ""
        source = ""

        forecast_file = live_dir / "live_forecast.npy"
        metadata_file = live_dir / "live_forecast_metadata.json"

        # Prefer the new AI forecast when the requested date matches.
        if forecast_file.exists() and metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            if (
                metadata.get("status") == "generated"
                and metadata.get("is_current_forecast") is True
            ):
                forecast_dates = [
                    str(d)[:10]
                    for d in metadata.get("forecast_dates", [])
                ]

                if not date and forecast_dates:
                    date = forecast_dates[-1]

                if date in forecast_dates:
                    forecast = np.load(
                        forecast_file, allow_pickle=False
                    )

                    if (
                        forecast.ndim != 4
                        or forecast.shape[0] != len(forecast_dates)
                        or forecast.shape[-1] != 3
                    ):
                        raise HTTPException(
                            status_code=500,
                            detail="Generated forecast has an unexpected shape."
                        )

                    index = forecast_dates.index(date)
                    values = np.asarray(
                        forecast[index, :, :, variables[variable]],
                        dtype=np.float64
                    )
                    prediction_date = date
                    source = "live_compatible_convlstm_forecast"

        # Preserve the existing historical predictions as a fallback.
        if values is None:
            dates_path = prepared_dir / "target_dates_test.npy"

            if not dates_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail="Historical prediction dates file not found."
                )

            dates = np.load(dates_path, allow_pickle=True)
            date_strings = [
                str(d.decode("utf-8") if isinstance(d, bytes) else d)[:10]
                for d in dates
            ]

            if not date_strings:
                raise HTTPException(
                    status_code=404,
                    detail="No historical prediction dates are available."
                )

            if date:
                if date not in date_strings:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No saved prediction is available for {date}."
                    )
                index = date_strings.index(date)
            else:
                index = len(date_strings) - 1

            prediction_file = evaluation_dir / f"pred_{variable}.npy"

            if not prediction_file.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Historical predictions not found for {variable}."
                )

            predictions = np.load(
                prediction_file, allow_pickle=False
            )

            if index >= len(predictions):
                raise HTTPException(
                    status_code=500,
                    detail="Historical prediction index is out of range."
                )

            values = np.asarray(predictions[index], dtype=np.float64).squeeze()
            prediction_date = date_strings[index]
            source = "historical_test_prediction"

    except HTTPException:
        raise
    except (OSError, ValueError, IndexError, KeyError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load prediction data: {exc}"
        ) from exc

    if values.shape != (len(latitudes), len(longitudes)):
        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction grid and coordinate dimensions do not match: "
                f"values={values.shape}, "
                f"coordinates=({len(latitudes)}, {len(longitudes)})"
            )
        )

    # Rainfall cannot be negative.
    if variable == "rainfall":
        values = np.maximum(values, 0.0)

    safe_values = [
        [
            float(value) if np.isfinite(value) else None
            for value in row
        ]
        for row in values
    ]

    return {
        "variable": variable,
        "prediction_date": prediction_date,
        "units": "mm" if variable == "rainfall" else "°C",
        "source": source,
        "latitudes": latitudes.tolist(),
        "longitudes": longitudes.tolist(),
        "values": safe_values,
    }


@app.get("/api/comparison-dates")
def get_comparison_dates():
    """Return dates available in the dedicated ConvLSTM comparison forecast."""
    project_root = Path(__file__).resolve().parent.parent
    live_dir = project_root / "outputs" / "live_prediction"
    forecast_file = live_dir / "comparison_forecast.npy"
    metadata_file = live_dir / "comparison_forecast_metadata.json"

    if not forecast_file.is_file() or not metadata_file.is_file():
        raise HTTPException(
            status_code=404,
            detail="Comparison forecast not generated yet."
        )

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if metadata.get("status") != "generated":
            raise HTTPException(
                status_code=409,
                detail="Comparison forecast is not marked as generated."
            )

        return {"dates": metadata.get("forecast_dates", [])}

    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read comparison metadata: {exc}"
        ) from exc


@app.get("/api/comparison-grid")
def get_comparison_grid(variable: str = "tmax", date: str = ""):
    """Return one date and variable from the dedicated ConvLSTM forecast."""
    aliases = {
        "max_temperature": "tmax",
        "min_temperature": "tmin",
    }
    variable = aliases.get(variable, variable)
    variable_indices = {"tmax": 0, "tmin": 1, "rainfall": 2}

    if variable not in variable_indices:
        raise HTTPException(
            status_code=400,
            detail="variable must be tmax, tmin, or rainfall"
        )

    project_root = Path(__file__).resolve().parent.parent
    live_dir = project_root / "outputs" / "live_prediction"
    forecast_file = live_dir / "comparison_forecast.npy"
    metadata_file = live_dir / "comparison_forecast_metadata.json"
    normalization_file = (
        project_root / "models" / "trained"
        / "live_compatible_normalization.npz"
    )
    spatial_dir = project_root / "outputs" / "spatial_maps"

    if not forecast_file.is_file() or not metadata_file.is_file():
        raise HTTPException(
            status_code=404,
            detail="Comparison forecast has not been generated."
        )

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        dates = [str(d)[:10] for d in metadata.get("forecast_dates", [])]

        if not dates:
            raise HTTPException(
                status_code=404,
                detail="No comparison forecast dates are available."
            )

        selected_date = date or dates[0]

        if selected_date not in dates:
            raise HTTPException(
                status_code=404,
                detail=f"No comparison forecast for {selected_date}."
            )

        forecast = np.load(forecast_file)
        expected_shape = (len(dates), 65, 73, 3)

        if forecast.shape != expected_shape:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Comparison forecast shape {forecast.shape} "
                    f"does not match expected {expected_shape}."
                )
            )

        norm = np.load(normalization_file)
        mask = norm["mask"].astype(bool)

        latitudes = np.asarray(
            np.load(spatial_dir / "latitude.npy")
        ).squeeze()
        longitudes = np.asarray(
            np.load(spatial_dir / "longitude.npy")
        ).squeeze()

        if mask.shape != (65, 73):
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected comparison mask shape: {mask.shape}"
            )

        if latitudes.size != 65 or longitudes.size != 73:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Spatial coordinate dimensions do not match "
                    "the comparison forecast grid."
                )
            )

        day_index = dates.index(selected_date)
        variable_index = variable_indices[variable]
        values = forecast[day_index, :, :, variable_index].copy()

        # Never display values outside the trained geographic mask.
        values[~mask] = np.nan

        if not np.isfinite(values[mask]).all():
            raise HTTPException(
                status_code=500,
                detail="Forecast contains invalid values inside the mask."
            )

        return {
            "variable": variable,
            "prediction_date": selected_date,
            "source": "ConvLSTM AI comparison forecast",
            "units": "mm" if variable == "rainfall" else "°C",
            "latitudes": latitudes.tolist(),
            "longitudes": longitudes.tolist(),
            "values": [
                [
                    float(v) if np.isfinite(v) else None
                    for v in row
                ]
                for row in values
            ],
        }

    except HTTPException:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load comparison forecast: {exc}"
        ) from exc

@app.get("/api/live-grid", summary="Maharashtra Live Weather Grid")
def get_live_grid() -> Dict[str, Any]:
    """Serve the prebuilt Open-Meteo grid for the dashboard."""
    live_grid_file = DATA_LIVE_DIR / "dashboard" / "live_weather.json"

    if not live_grid_file.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                "Live weather grid not found. Run "
                "src/data/build_live_dashboard_data.py first."
            ),
        )

    try:
        with open(live_grid_file, "r", encoding="utf-8") as file:
            payload = json.load(file)

        if not payload.get("dates") or not payload.get("grid"):
            raise HTTPException(
                status_code=500,
                detail="Live weather grid is missing dates or coordinates.",
            )

        return payload

    except HTTPException:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Could not read live weather grid: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Could not read the live weather grid.",
        ) from exc

    
@app.get("/api/risk-indicators", summary="Experimental Climate Risk Indicators")
def get_risk_indicators():
    """
    Calculate transparent rainfall-threshold indicators and temperature
    summaries from existing model forecasts. These are not official warnings.
    """
    project_root = Path(__file__).resolve().parent.parent
    forecast_dir = project_root / "outputs" / "live_prediction"
    mask_path = (
        project_root / "data" / "processed_daily" / "cnn_dataset"
        / "multivariable_prepared" / "spatial_mask.npy"
    )

    if not mask_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Maharashtra spatial mask is missing."
        )

    mask = np.load(mask_path).astype(bool)

    if mask.ndim != 2:
        raise HTTPException(
            status_code=500,
            detail="Spatial mask must be a 2D array."
        )

    # Prefer the current live forecast; use comparison only as a fallback.
    candidates = [
        ("comparison_forecast.npy", "comparison_forecast_metadata.json"),
        ("live_forecast.npy", "live_forecast_metadata.json"),
    ]

    forecast = None
    metadata = None
    source = None

    for array_name, metadata_name in candidates:
        array_path = forecast_dir / array_name
        metadata_path = forecast_dir / metadata_name

        if not array_path.exists() or not metadata_path.exists():
            continue

        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                candidate_metadata = json.load(f)

            candidate = np.load(array_path, allow_pickle=False)

            if (
                candidate.ndim == 4
                and candidate.shape[1:3] == mask.shape
                and candidate.shape[-1] == 3
                and len(candidate_metadata.get("forecast_dates", []))
                    == candidate.shape[0]
            ):
                forecast = candidate
                metadata = candidate_metadata
                source = array_name
                break
        except (OSError, ValueError, json.JSONDecodeError):
            continue

    if forecast is None:
        raise HTTPException(
            status_code=404,
            detail="No usable forecast and matching metadata were found."
        )

    # Daily rainfall thresholds (mm) used by IMD rainfall categories.
    # These classify model output; they do not establish an official warning.
    results = []

    for day_index, day in enumerate(metadata["forecast_dates"]):
        day_grid = forecast[day_index]
        tmax = day_grid[:, :, 0][mask]
        tmin = day_grid[:, :, 1][mask]
        rain = day_grid[:, :, 2][mask]

        valid_tmax = tmax[np.isfinite(tmax)]
        valid_tmin = tmin[np.isfinite(tmin)]
        valid_rain = rain[np.isfinite(rain)]

        def stats(values, unit):
            if values.size == 0:
                return {
                    "status": "insufficient_data",
                    "unit": unit,
                    "valid_cells": 0,
                }

            return {
                "status": "available",
                "unit": unit,
                "valid_cells": int(values.size),
                "mean": round(float(np.mean(values)), 2),
                "minimum": round(float(np.min(values)), 2),
                "maximum": round(float(np.max(values)), 2),
            }

        rainfall_categories = {
            "status": (
                "available" if valid_rain.size else "insufficient_data"
            ),
            "unit": "grid_cells",
            "heavy_64_5_to_115_5_mm": int(
                np.sum((valid_rain >= 64.5) & (valid_rain < 115.6))
            ),
            "very_heavy_115_6_to_204_4_mm": int(
                np.sum((valid_rain >= 115.6) & (valid_rain < 204.5))
            ),
            "extremely_heavy_204_5_mm_or_more": int(
                np.sum(valid_rain >= 204.5)
            ),
        }

        results.append({
            "date": str(day)[:10],
            "temperature": {
                "tmax": stats(valid_tmax, "degC"),
                "tmin": stats(valid_tmin, "degC"),
                "heatwave_classification": (
                    "not_assessed: requires applicable regional criteria "
                    "and climatological normals"
                ),
            },
            "rainfall": {
                **stats(valid_rain, "mm"),
                "threshold_indicators": rainfall_categories,
            },
        })

    return {
        "status": "experimental",
        "source_file": source,
        "forecast_type": metadata.get("forecast_type", "model_forecast"),
        "forecast_dates": metadata["forecast_dates"],
        "spatial_mask": "spatial_mask.npy",
        "valid_spatial_cells": int(mask.sum()),
        "official_warning": False,
        "notice": (
            "These are model-derived indicators, not verified observations "
            "or official warnings. Validate forecasts before operational use."
        ),
        "results": results,
    }


@app.post("/api/what-if")
def run_what_if(payload: Dict[str, Any]):
    """
    Experimental post-processing scenario.
    Does not rerun the trained model or modify official forecast files.
    """
    project_root = Path(__file__).resolve().parent.parent
    forecast_dir = project_root / "outputs" / "live_prediction"
    mask_path = (
        project_root / "data" / "processed_daily" / "cnn_dataset"
        / "multivariable_prepared" / "spatial_mask.npy"
    )

    try:
        deltas = {
            "tmax_c": float(payload.get("tmax_delta", 0)),
            "tmin_c": float(payload.get("tmin_delta", 0)),
            "rainfall_mm": float(payload.get("rainfall_delta", 0)),
        }
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="Scenario changes must be numeric.")

    limits = {
        "tmax_c": (-10, 10),
        "tmin_c": (-10, 10),
        "rainfall_mm": (-100, 100),
    }
    for name, value in deltas.items():
        low, high = limits[name]
        if not np.isfinite(value) or not low <= value <= high:
            raise HTTPException(
                status_code=422,
                detail=f"{name} change must be between {low} and {high}.",
            )

    if not mask_path.exists():
        raise HTTPException(status_code=500, detail="Spatial mask is missing.")

    mask = np.load(mask_path).astype(bool)
    forecast = None
    metadata = None
    source = None

    # Use the same forecast-source preference as the existing risk endpoint.
    for array_name, metadata_name in [
        ("comparison_forecast.npy", "comparison_forecast_metadata.json"),
        ("live_forecast.npy", "live_forecast_metadata.json"),
    ]:
        array_path = forecast_dir / array_name
        metadata_path = forecast_dir / metadata_name

        if not array_path.exists() or not metadata_path.exists():
            continue

        try:
            candidate = np.load(array_path, allow_pickle=False)
            candidate_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                candidate.ndim == 4
                and candidate.shape[1:3] == mask.shape
                and candidate.shape[-1] == 3
                and len(candidate_meta.get("forecast_dates", [])) == candidate.shape[0]
            ):
                forecast, metadata, source = candidate, candidate_meta, array_name
                break
        except (OSError, ValueError, json.JSONDecodeError):
            continue

    if forecast is None:
        raise HTTPException(
            status_code=404,
            detail="No compatible forecast is available. Generate a forecast first.",
        )

    # Copy in memory only. Never overwrite the official forecast.
    scenario = forecast.astype(np.float32, copy=True)
    for channel, name in enumerate(["tmax_c", "tmin_c", "rainfall_mm"]):
        scenario[..., channel] += deltas[name]

    # Rainfall cannot be negative; exclude cells outside the spatial mask.
    scenario[..., 2] = np.maximum(scenario[..., 2], 0)
    scenario[:, ~mask, :] = np.nan

    def summarize(array):
        results = []
        for index, date in enumerate(metadata["forecast_dates"]):
            day = array[index]
            item = {"date": date}
            for channel, name, unit in [
                (0, "tmax", "°C"),
                (1, "tmin", "°C"),
                (2, "rainfall", "mm/day"),
            ]:
                values = day[..., channel]
                values = values[np.isfinite(values)]
                item[name] = {
                    "mean": round(float(np.mean(values)), 3) if values.size else None,
                    "min": round(float(np.min(values)), 3) if values.size else None,
                    "max": round(float(np.max(values)), 3) if values.size else None,
                    "unit": unit,
                }
            results.append(item)
        return results

    return {
        "status": "experimental_scenario",
        "method": "post_processing_sensitivity",
        "source_file": source,
        "forecast_dates": metadata["forecast_dates"],
        "changes": deltas,
        "baseline": summarize(forecast),
        "scenario": summarize(scenario),
        "notice": (
            "Hypothetical sensitivity analysis only. The trained model was not rerun; "
            "these changes are applied to forecast outputs. Not an official warning."
        ),
    }
