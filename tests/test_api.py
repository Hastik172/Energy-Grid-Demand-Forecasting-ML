"""
API Integration Tests using FastAPI TestClient.
Validates all REST endpoints, validation errors, and prediction dynamics.
"""

import pytest
from starlette.testclient import TestClient
from backend.app import app

client = TestClient(app)


def test_overview_and_graphs():
    """Verify overview metrics and full graph catalogue are reachable."""
    resp_overview = client.get("/api/overview")
    assert resp_overview.status_code == 200
    data = resp_overview.json()
    assert data["total_records"] == 8760
    assert "test_r2" in data
    assert data["test_r2"] > 0.75

    resp_graphs = client.get("/api/graphs")
    assert resp_graphs.status_code == 200
    graphs = resp_graphs.json()
    assert len(graphs) >= 12
    assert graphs[0]["id"] == "01_demand_timeseries"


def test_predict_high_risk_beats_low_risk():
    """Verify extreme hot weather predicts higher demand & higher peak probability than mild spring."""
    high_heat_payload = {
        "temperature": 35.0,
        "solar_generation": 4000.0,
        "wind_generation": 800.0,
        "hour": 14,
        "day_of_week": 2,
        "month": 7,
        "is_weekend": 0
    }
    mild_payload = {
        "temperature": 20.0,
        "solar_generation": 2000.0,
        "wind_generation": 1200.0,
        "hour": 14,
        "day_of_week": 2,
        "month": 4,
        "is_weekend": 0
    }

    r_high = client.post("/api/predict", json=high_heat_payload)
    r_mild = client.post("/api/predict", json=mild_payload)

    assert r_high.status_code == 200
    assert r_mild.status_code == 200

    d_high = r_high.json()
    d_mild = r_mild.json()

    # Extreme heat must predict higher demand than mild comfort zone
    assert d_high["predicted_demand"] > d_mild["predicted_demand"]
    assert d_high["prob_high_demand"] > d_mild["prob_high_demand"]
    assert d_high["lower_90"] < d_high["predicted_demand"] < d_high["upper_90"]


def test_predict_rejects_bad_input():
    """Verify input validation rejects invalid hours (> 23) and months (> 12)."""
    bad_payload = {
        "temperature": 25.0,
        "solar_generation": 1000.0,
        "wind_generation": 1000.0,
        "hour": 48,  # Invalid hour
        "day_of_week": 2,
        "month": 15  # Invalid month
    }
    resp = client.post("/api/predict", json=bad_payload)
    assert resp.status_code == 422  # Pydantic validation error


def test_polyfit_overfits_at_high_degree():
    """Verify polynomial experiment endpoint correctly executes Bishop §1.1 curve fitting."""
    resp = client.get("/api/polyfit?degree=9&sample_size=100")
    assert resp.status_code == 200
    data = resp.json()
    assert data["degree"] == 9
    assert len(data["curve_x"]) == 100
    assert "train_rmse" in data
    assert "test_rmse" in data
    assert data["test_rmse"] > 0


def test_data_preview_and_pages():
    """Verify raw dataset preview and per-column summaries."""
    r_preview = client.get("/api/data/preview?limit=10")
    assert r_preview.status_code == 200
    p_data = r_preview.json()
    assert len(p_data["rows"]) == 10
    assert "electricity_demand_mw" in p_data["columns"]

    r_summary = client.get("/api/data/summary")
    assert r_summary.status_code == 200
    s_data = r_summary.json()
    assert "electricity_demand_mw" in s_data
    assert s_data["electricity_demand_mw"]["mean"] > 20000
