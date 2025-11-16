import pytest
from backend.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_and_cleanup():
    """Menjalankan sebelum dan sesudah semua test."""
    # sebelum test
    yield
    # sesudah test — hapus semua data
    client.delete("/api/v1/data/delete-all")

def test_get_all_data_returns_json():
    res = client.get("/api/v1/data")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

def test_get_info_returns_structure():
    res = client.get("/api/v1/data/info")
    assert res.status_code == 200
    body = res.json()
    for key in ["totalData", "outlierClear", "nanClear"]:
        assert key in body

def test_delete_all_data():
    res = client.delete("/api/v1/data/delete-all")
    assert res.status_code == 200
    assert "berhasil" in res.json()["message"]

def test_outlier_detection_and_handle(monkeypatch):
    """Mock data untuk menguji outlier detection"""
    import pandas as pd
    from helpers import detect_outliers

    # buat data sederhana dengan outlier
    df = pd.DataFrame({
        "id": [1,2,3],
        "pm10": [10, 12, 999],
        "pm25": [1, 1, 1],
        "so2": [2, 2, 2],
        "co": [3, 3, 3],
        "o3": [4, 4, 4],
        "no2": [5, 5, 5],
        "hc": [6, 6, 6],
        "kelembaban": [7, 7, 7],
        "suhu": [8, 8, 8]
    })
    outliers = detect_outliers(df)
    assert any(o["Kolom"] == "pm10" for o in outliers)
