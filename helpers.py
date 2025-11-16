import pandas as pd
import numpy as np
import traceback
from fastapi.responses import JSONResponse
from db import get_db_connection

def fetch_all_data():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SHOW TABLES LIKE 'air_quality_data'")
        if not cursor.fetchone():
            # Jika tabel belum ada
            print("⚠️ Table 'air_quality_data' not found. Returning empty dataset.")
            return []

        cursor.execute("SELECT * FROM air_quality_data ORDER BY waktu ASC")
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        return rows

    except Exception as e:
        print("❌ Error in fetch_all_data():", e)
        traceback.print_exc()
        return []

# def detect_outliers(df: pd.DataFrame):
#     numeric_cols = ["pm10","pm25","so2","co","o3","no2","hc","kelembaban","suhu"]
#     outliers = []

#     for col in numeric_cols:
#         if col not in df.columns:
#             continue
#         mean = df[col].mean()
#         std = df[col].std()
#         mask = (df[col] - mean).abs() > 3*std
#         for idx in df[mask].index:
#             outliers.append({
#                 "id": int(df.loc[idx, "id"]),
#                 "Kolom": col,
#                 "Nilai": df.loc[idx, col]
#             })
#     return outliers

def detect_outliers(df: pd.DataFrame):
    """
    Deteksi outlier adaptif:
    - Jika jumlah data kecil (<10): deteksi nilai ekstrem berbasis rasio median
    - Jika jumlah data besar: gunakan metode IQR
    """
    numeric_cols = ["pm10", "pm25", "so2", "co", "o3", "no2", "hc", "kelembaban", "suhu"]
    outliers = []

    for col in numeric_cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()

        if len(series) == 0:
            continue

        # Mode 1: dataset kecil (<10) → gunakan rasio median
        if len(series) < 10:
            median_val = np.median(series)
            if median_val == 0:
                median_val = 1e-6  # hindari pembagian nol
            ratio_threshold = 3.0  # lebih ketat dari sebelumnya
            mask = (series > median_val * ratio_threshold) | (series < median_val / ratio_threshold)

        # Mode 2: dataset besar → gunakan IQR
        else:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (series < lower) | (series > upper)

        for idx in series[mask].index:
            outliers.append({
                "id": int(df.loc[idx, "id"]),
                "Kolom": col,
                "Nilai": float(df.loc[idx, col])
            })

    return outliers

def get_data_info(rows):
    if not rows:
        return {
            "totalData": 0,
            "outlierClear": True,
            "nanClear": True,
            "outlierCount": 0,
            "nanCount": 0
        }

    df = pd.DataFrame(rows)
    total_data = len(df)
    nan_clear = not df.isnull().values.any()
    nan_count = int(df.isnull().sum().sum())
    numeric_cols = ["pm10","pm25","so2","co","o3","no2","hc","kelembaban","suhu"]
    outlier_count = int(df[numeric_cols].apply(lambda x: ((x - x.mean()).abs() > 3*x.std()).sum()).sum())

    return {
        "totalData": total_data,
        "outlierClear": outlier_count == 0,
        "nanClear": nan_clear,
        "outlierCount": outlier_count,
        "nanCount": nan_count
    }
