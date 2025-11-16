# Backend (AirQ) — Dokumentasi

> Backend API untuk proyek AirQ, dibangun dengan FastAPI. Menyediakan endpoint untuk prediksi kualitas udara (Prophet), manajemen data CSV, deteksi & penanganan outlier, dan operasi CRUD sederhana terhadap tabel MySQL.

## Ringkasan

File utama: `main.py` (terletak di folder `backend/`).

## Persyaratan

uvicorn[standard]
pandas
numpy
prophet
optuna
holidays
scikit-learn
mysql-connector-python
python-multipart

````

## Persiapan (Windows PowerShell)

1. Buka PowerShell di folder `backend`.
2. Buat virtual environment (jika belum ada):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
````

3. Install dependency:

```powershell
pip install -r requirements.txt
```

Catatan: di repository sudah terdapat `venv_msi/` — gunakan virtualenv yang Anda pilih atau environment tersebut jika sudah sesuai.

## Konfigurasi database

Di `main.py` terdapat helper `get_db_connection()` yang saat ini terkonfigurasi secara statis:

```py
mysql.connector.connect(
    host="localhost",
    user="root",
    password="",  # ubah sesuai user/password kamu
    database="db_airq"  # ubah sesuai nama DB kamu
)
```

Ubah nilai `host`, `user`, `password`, dan `database` sesuai lingkungan Anda. Untuk production, simpan kredensial ini di environment variable dan muat dari sana.

### Skema tabel (minimal yang diperlukan)

Backend mengakses dua tabel berbeda dalam beberapa bagian kode:

1. `air_quality` — dipakai saat `get_all_data()` (SELECT \* FROM air_quality ...). Struktur tidak kritis untuk dokumentasi ini, tapi harus memiliki kolom `waktu` yang berformat DATETIME atau TIMESTAMP.

2. `air_quality_data` — tabel utama untuk CRUD dan upload CSV. Kolom yang digunakan:

- id (INT, primary key, auto increment)
- waktu (DATETIME)
- pm10 (FLOAT)
- pm25 (FLOAT)
- so2 (FLOAT)
- co (FLOAT)
- o3 (FLOAT)
- no2 (FLOAT)
- hc (FLOAT)
- kelembaban (FLOAT, nullable)
- suhu (FLOAT, nullable)

Contoh DDL sederhana (MySQL):

```sql
CREATE TABLE air_quality_data (
  id INT AUTO_INCREMENT PRIMARY KEY,
  waktu DATETIME NOT NULL,
  pm10 DOUBLE,
  pm25 DOUBLE,
  so2 DOUBLE,
  co DOUBLE,
  o3 DOUBLE,
  no2 DOUBLE,
  hc DOUBLE,
  kelembaban DOUBLE,
  suhu DOUBLE
);
```

## Menjalankan server (development)

Jalankan Uvicorn dari folder `backend`:

```powershell
# aktifkan virtualenv terlebih dahulu
.\\.venv\\Scripts\\Activate.ps1
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Sekarang API akan tersedia di `http://127.0.0.1:8000` dan dokumentasi interaktif Swagger di `http://127.0.0.1:8000/docs`.

## Endpoints utama

- GET `/api/v1/air-quality` — prediksi kualitas udara untuk tanggal saat ini per polutan (PM10, PM25, SO2, CO, O3, NO2, HC)
- GET `/api/v1/forecast` — pratinjau prediksi beberapa hari ke depan (hasil per polutan)
- GET `/api/v1/predict/{date}` — prediksi untuk tanggal tertentu (format YYYY-MM-DD)
- GET `/api/v1/mape` — perhitungan MAPE (menggunakan cross-validation Prophet)
- GET `/api/data` — ambil semua data dari tabel `air_quality` (atau endpoint berbeda yang mengembalikan seluruh dataset)
- GET `/api/v1/data` — ambil seluruh baris dari `air_quality_data` (terurut DESC pada waktu)
- GET `/api/v1/data/info` — ringkasan info dataset: totalData, outlierCount, nanCount, dll.
- GET `/api/v1/data/outliers` — daftar outlier yang terdeteksi
- POST `/api/v1/data/outliers-handle` — jalankan proses interpolasi untuk menangani outlier
- POST `/api/v1/input` — simpan satu baris data (JSON) ke `air_quality_data`
- POST `/api/v1/upload-csv` — unggah file CSV (multipart/form-data) dan masukkan ke DB
- DELETE `/api/v1/data/delete-all` — hapus semua data di `air_quality_data`

Catatan: semua endpoint mengembalikan JSON. Untuk upload CSV gunakan form-data dengan key `file`.

## Format CSV yang disarankan untuk upload

Kolom wajib: `Waktu`, `PM10`, `PM25`, `SO2`, `CO`, `O3`, `NO2`, `HC`
Kolom opsional: `Kelembaban`, `Suhu`

`Waktu` akan diparse dengan `pd.to_datetime(...)` — pastikan format dapat diparse (mis. `YYYY-MM-DD HH:MM:SS`). Baris yang gagal parse waktu akan diabaikan (dihapus).

## Penanganan outlier

Backend mendeteksi outlier menggunakan aturan sederhana (> 3 std dev dari mean) untuk kolom numerik. Endpoint `outliers-handle` akan mengubah nilai outlier menjadi NaN dan melakukan interpolasi linear untuk mengisi nilai.

## Troubleshooting

- Error koneksi DB: periksa `get_db_connection()` dan MySQL server berjalan, kredensial benar, dan user punya hak INSERT/SELECT/UPDATE.
- Masalah instalasi Prophet: library `prophet` membutuhkan compiler dan beberapa dependency; jika terjadi error, pastikan Python dev tools terpasang dan gunakan wheel prebuilt bila perlu.
- CSV upload gagal: periksa encoding (UTF-8) dan nama kolom sesuai yang diharapkan.

## Catatan pengembangan

- Untuk production, pindahkan konfigurasi sensitif ke environment variables.
- Cache model Prophet disimpan di memori (`model_cache`) saat server hidup; restart server akan mengosongkan cache.

## Kontak dan kontribusi

Jika ingin menambah fitur, buka issue/PR di repository. Sertakan contoh CSV bila berhubungan dengan data.
