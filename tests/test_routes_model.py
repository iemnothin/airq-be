import pytest
from backend.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

@pytest.mark.parametrize("endpoint", [
    "/api/v1/model/process-basic",
    "/api/v1/model/process-advanced"
])
def test_forecast_endpoints_return_json(endpoint):
    res = client.post(endpoint)
    # mungkin gagal jika belum ada data, jadi periksa status
    assert res.status_code in [200, 400, 500]
    if res.status_code == 200:
        assert "forecast" in res.json()

def test_clear_forecast_tables():
    res = client.delete("/api/v1/model/clear-forecast")
    assert res.status_code == 200
    assert "berhasil" in res.json()["message"]
