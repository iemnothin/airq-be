import io
from backend.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_upload_csv_success(monkeypatch):
    csv_content = (
        "Waktu,PM10,PM25,SO2,CO,O3,NO2,HC,Kelembaban,Suhu\n"
        "2024-07-30,74,15,6,351,33,17,388,72,30\n"
        "2024-07-31,64,19,8,253,38,53,353,78,28\n"
    )
    file = io.BytesIO(csv_content.encode("utf-8"))
    files = {"file": ("test.csv", file, "text/csv")}

    response = client.post("/api/v1/upload-csv", files=files)
    assert response.status_code in [200, 201]
    assert "Upload berhasil" in response.text

def test_upload_csv_invalid_extension():
    fake_txt = io.BytesIO(b"dummy text")
    files = {"file": ("test.txt", fake_txt, "text/plain")}
    res = client.post("/api/v1/upload-csv", files=files)
    assert res.status_code == 400
    assert "harus berformat CSV" in res.text

def test_upload_csv_missing_column():
    bad_csv = "Waktu,PM10,PM25,SO2\n2024-07-30,74,15,6\n"
    file = io.BytesIO(bad_csv.encode("utf-8"))
    files = {"file": ("bad.csv", file, "text/csv")}
    res = client.post("/api/v1/upload-csv", files=files)
    assert res.status_code == 400
