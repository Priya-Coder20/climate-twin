"""
Unit and integration tests for the AI Digital Twin Backend.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import pytest
from fastapi.testclient import TestClient
from backend.main import app, PROJECT_ROOT, EXPECTED_CHANNELS, REQUIRED_INPUT_DAYS

@pytest.fixture
def client():
    return TestClient(app)

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "health" in data["endpoints"]
    assert "status" in data["endpoints"]
    assert "prediction" in data["endpoints"]

def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["data_live_directory"]["exists"] is True
    assert data["outputs_directory"]["exists"] is True

def test_status_endpoint_dynamic_discovery(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()

    # Verify providers have dates discovered dynamically (not hardcoded)
    dates = data["latest_available_dates"]
    assert dates["mosdac"] is not None
    assert dates["era5"] is not None
    assert dates["imd"] is not None

    # Verify latest common date is computed
    common = data["latest_common_date"]
    assert "processed_channels_all_10" in common
    assert common["processed_channels_all_10"] is not None

    # Verify channel expectations
    assert len(data["expected_channels"]) == 10
    for ch in EXPECTED_CHANNELS:
        assert ch in data["expected_channels"]

    # Verify forecast feasibility
    feasibility = data["forecast_feasibility"]
    assert "new_data_available" in feasibility
    assert "is_stale" in feasibility
    assert "is_sufficient_for_forecast" in feasibility
    assert isinstance(feasibility["is_sufficient_for_forecast"], bool)
    assert len(feasibility["reason"]) > 0

    # Verify consecutive input window
    window = data["consecutive_input_window"]
    assert window["required_days"] == REQUIRED_INPUT_DAYS
    assert isinstance(window["is_window_complete"], bool)

def test_prediction_endpoint_non_fabricated(client):
    # Default call should not fabricate predictions
    response = client.get("/api/prediction")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is False
    assert data["status"] == "unavailable"
    assert data["prediction_date"] is None
    assert data["tmax"] is None
    assert data["tmin"] is None
    assert "alignment_summary" in data

    # Historical evaluation artifact inspection
    if data["verified_disk_artifact"] is not None:
        assert data["verified_disk_artifact"]["available"] is True
        assert "tmax" in data["verified_disk_artifact"]
        assert "tmin" in data["verified_disk_artifact"]

def test_prediction_endpoint_historical_artifact_mode(client):
    response = client.get("/api/prediction?include_historical=true")
    assert response.status_code == 200
    data = response.json()
    if data.get("verified_disk_artifact"):
        assert data["available"] is True
        assert data["status"] == "historical_artifact"
        assert data["prediction_date"] is not None
        assert data["tmax"]["mean"] is not None
        assert data["tmin"]["mean"] is not None

def test_explainability_features(client):
    response = client.get("/api/explainability/features")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "tmax" in data
    assert "tmin" in data
    assert isinstance(data["tmax"], dict)
    assert len(data["tmax"]) > 0

def test_spatial_maps(client):
    # Test valid tmax map
    res_tmax = client.get("/api/maps/tmax")
    assert res_tmax.status_code == 200
    assert "image/png" in res_tmax.headers.get("content-type", "")
    assert len(res_tmax.content) > 1000

    # Test valid tmin map
    res_tmin = client.get("/api/maps/tmin")
    assert res_tmin.status_code == 200
    assert "image/png" in res_tmin.headers.get("content-type", "")
    assert len(res_tmin.content) > 1000

def test_spatial_maps_invalid_type(client):
    # Test invalid map type returns 400
    res_invalid = client.get("/api/maps/rainfall")
    assert res_invalid.status_code == 400
    assert "detail" in res_invalid.json()

def test_parse_mosdac_raw_date_helper():
    from backend.main import parse_mosdac_raw_date
    assert parse_mosdac_raw_date("3RIMG_14SEP2026_0015_L3C_PET_DLY_V01R00.h5") == "2026-09-14"
    assert parse_mosdac_raw_date("3RIMG_01JAN2025_0015_L3G_IMR_DLY_V01R00.h5") == "2025-01-01"
    assert parse_mosdac_raw_date("non_matching_filename.txt") is None

def test_missing_directory_graceful_handling(tmp_path):
    from backend.main import inspect_system_status
    # If passed a non-existent or empty directory
    fake_root = tmp_path / "empty_project"
    fake_root.mkdir()
    result = inspect_system_status(fake_root)
    assert result["status"] == "missing_directory"
    assert result["is_sufficient_for_forecast"] is False

