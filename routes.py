import io
import traceback
import time
import json
from fastapi.responses import StreamingResponse
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import JSONResponse
import pandas as pd
from db import get_db_connection
from helpers import fetch_all_data, get_data_info, detect_outliers
from ml import (
    process_basic_forecast,
    process_advanced_forecast_stream
)

router = APIRouter(prefix="/api/v1", tags=["Air Quality Data & Forecasting"])

# ============================================================
# 1. GET All Air Quality Data
# ============================================================
@router.get("/data")
def get_all_data():
    try:
        rows = fetch_all_data()

        # 🔥 FIX — convert datetime to JSON-serializable string
        def serialize_datetime(obj):
            for k, v in obj.items():
                if isinstance(v, datetime):
                    obj[k] = v.strftime("%Y-%m-%d %H:%M:%S")
            return obj

        rows = [serialize_datetime(dict(r)) for r in rows]
        return JSONResponse(content={"data": rows, "status": "ok"}, status_code=200)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ============================================================
# 2. DATA INFO
# ============================================================
@router.get("/data/info")
def get_info():
    try:
        rows = fetch_all_data()
        if not rows:
            default_info = {
                "totalData": 0,
                "outlierClear": True,
                "nanClear": True,
                "outlierCount": 0,
                "nanCount": 0,
                "message": "No data available yet.",
            }
            return JSONResponse(content=default_info, status_code=200)

        info = get_data_info(rows)
        return JSONResponse(content=info, status_code=200)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ============================================================
# 3. GET Outliers
# ============================================================
@router.get("/data/outliers")
def get_outliers():
    try:
        rows = fetch_all_data()
        if not rows:
            return JSONResponse(content={"outliers": [], "status": "ok"}, status_code=200)

        df = pd.DataFrame(rows)
        outliers = detect_outliers(df)
        return JSONResponse(content={"outliers": outliers, "status": "ok"}, status_code=200)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ============================================================
# 4. Handle Outliers (Interpolate)
# ============================================================
@router.post("/data/outliers-handle")
def handle_outliers():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM air_quality_data ORDER BY waktu ASC")
        rows = cursor.fetchall()

        if not rows:
            cursor.close()
            conn.close()
            return JSONResponse(content={"message": "No data available."}, status_code=200)

        df = pd.DataFrame(rows)
        df["waktu"] = pd.to_datetime(df["waktu"])
        numeric_cols = ["pm10", "pm25", "so2", "co", "o3", "no2", "hc", "kelembaban", "suhu"]

        outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)
        for col in numeric_cols:
            mean, std = df[col].mean(), df[col].std()
            outlier_mask[col] = (df[col] - mean).abs() > 3 * std

        if not outlier_mask.values.any():
            cursor.close()
            conn.close()
            return JSONResponse(content={"message": "No outliers detected."}, status_code=200)

        for col in numeric_cols:
            df.loc[outlier_mask[col], col] = None
            df[col] = df[col].interpolate(method="linear", limit_direction="both")

        for _, row in df.iterrows():
            sql = """
                UPDATE air_quality_data
                SET pm10=%s, pm25=%s, so2=%s, co=%s, o3=%s, no2=%s, hc=%s,
                    kelembaban=%s, suhu=%s
                WHERE id=%s
            """
            cursor.execute(
                sql,
                (row["pm10"], row["pm25"], row["so2"], row["co"], row["o3"],
                 row["no2"], row["hc"], row["kelembaban"], row["suhu"], row["id"])
            )

        conn.commit()
        cursor.close()
        conn.close()

        return JSONResponse(content={
            "message": f"{int(outlier_mask.values.sum())} outlier values interpolated successfully."
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ============================================================
# 5. Delete All Data
# ============================================================
@router.delete("/data/delete-all")
def delete_all_data():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM air_quality_data")
        conn.commit()
        cursor.close()
        conn.close()
        return JSONResponse(content={"message": "All data records have been deleted successfully."})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ============================================================
# 6. Upload CSV
# ============================================================
@router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".csv"):
            return JSONResponse({"error": "File must be in CSV format."}, status_code=400)

        content = await file.read()
        data = pd.read_csv(io.StringIO(content.decode("utf-8")))

        required_cols = ["waktu", "pm10", "pm25", "so2", "co", "o3", "no2", "hc"]
        for col in required_cols:
            if col not in data.columns:
                return JSONResponse({"error": f"Missing column '{col}'"}, 400)

        if "kelembaban" not in data.columns:
            data["kelembaban"] = None
        if "suhu" not in data.columns:
            data["suhu"] = None

        data["waktu"] = pd.to_datetime(data["waktu"], errors="coerce")
        data = data.dropna(subset=["waktu"])

        conn = get_db_connection()
        cursor = conn.cursor()
        query = """
            INSERT INTO air_quality_data
            (waktu, pm10, pm25, so2, co, o3, no2, hc, kelembaban, suhu)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """

        for _, r in data.iterrows():
            cursor.execute(
                query,
                (
                    r["waktu"].strftime("%Y-%m-%d %H:%M:%S"),
                    r["pm10"], r["pm25"], r["so2"], r["co"],
                    r["o3"], r["no2"], r["hc"], r["kelembaban"], r["suhu"]
                )
            )

        conn.commit()
        cursor.close()
        conn.close()
        return JSONResponse({"message": "CSV file uploaded and saved successfully."})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# 7. Manual Data Input
# ============================================================
@router.post("/input")
async def input_air_quality(request: Request):
    try:
        data = await request.json()
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = """
        INSERT INTO air_quality_data
        (waktu, pm10, pm25, so2, co, o3, no2, hc, kelembaban, suhu)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        cursor.execute(
            sql,
            (
                data.get("waktu"), data.get("pm10"), data.get("pm25"),
                data.get("so2"), data.get("co"), data.get("o3"),
                data.get("no2"), data.get("hc"), data.get("kelembaban"), data.get("suhu")
            )
        )
        conn.commit()
        cursor.close()
        conn.close()
        return JSONResponse({"message": "Record inserted successfully into database."})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# 8. Process Forecast Basic
# ============================================================
@router.post("/model/process-basic")
def process_basic_all():
    try:
        rows = fetch_all_data()
        if not rows:
            return JSONResponse({"error": "No data available for processing."}, status_code=400)

        df = pd.DataFrame(rows)
        df["waktu"] = pd.to_datetime(df["waktu"])
        pollutants = ["pm10", "pm25", "so2", "o3", "no2", "co", "hc"]
        forecast = process_basic_forecast(df, pollutants)

        return JSONResponse({
            "message": "Forecast Prophet successfully processed for all pollutants.",
            "forecast": forecast,
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# 9. Process Forecast Advanced (Streaming)
# ============================================================
@router.post("/model/process-advanced")
def process_advanced_all():
    def progress_stream():
        try:
            rows = fetch_all_data()
            if not rows:
                yield f"data: {json.dumps({'status': 'error', 'message': 'No data available for processing.'})}\n\n"
                return

            df = pd.DataFrame(rows)
            df["waktu"] = pd.to_datetime(df["waktu"])
            pollutants = ["pm10", "pm25", "so2", "o3", "no2", "co", "hc"]
            total = len(pollutants)

            yield f"data: {json.dumps({'status': 'start', 'total': total, 'message': 'Starting advanced forecast...'})}\n\n"

            for idx, pol in enumerate(pollutants, start=1):
                try:
                    yield f"data: {json.dumps({'status': 'processing', 'pollutant': pol.upper(), 'progress': round((idx - 1) / total * 100, 2)})}\n\n"
                    process_advanced_forecast_stream(df, [pol])
                    yield f"data: {json.dumps({'status': 'done', 'pollutant': pol.upper(), 'progress': round(idx / total * 100, 2)})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'status': 'error', 'pollutant': pol.upper(), 'message': str(e)})}\n\n"
                    continue

                time.sleep(0.3)

            yield f"data: {json.dumps({'status': 'complete', 'progress': 100, 'message': 'All forecasts processed successfully!'})}\n\n"

        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(progress_stream(), media_type="text/event-stream")


# ============================================================
# 10. Clear Forecast Tables
# ============================================================
@router.delete("/model/clear-forecast")
def clear_forecast():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        tables = [
            "forecast_pm10_data", "forecast_pm10_with_parameters_data",
            "forecast_pm25_data", "forecast_pm25_with_parameters_data",
            "forecast_so2_data", "forecast_so2_with_parameters_data",
            "forecast_o3_data", "forecast_o3_with_parameters_data",
            "forecast_no2_data", "forecast_no2_with_parameters_data",
            "forecast_co_data", "forecast_co_with_parameters_data",
            "forecast_hc_data", "forecast_hc_with_parameters_data",
        ]
        for t in tables:
            cursor.execute(f"TRUNCATE TABLE {t}")

        conn.commit()
        cursor.close()
        conn.close()
        return JSONResponse({"message": "All forecast tables have been cleared successfully."})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# 11. SYSTEM STATUS & TECHNOLOGIES
# ============================================================
@router.get("/status")
def system_status():
    import psutil
    import shutil
    from datetime import datetime
    from db import get_db_connection

    try:
        conn = get_db_connection()
        if conn:
            conn.cursor().execute("SELECT 1")
            conn.close()
            db_status = "connected"
        else:
            db_status = "disconnected"
    except:
        db_status = "disconnected"

    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory().percent
    disk = shutil.disk_usage("/").percent if hasattr(shutil.disk_usage("/"), "percent") else 0

    if cpu < 80 and ram < 80 and disk < 90:
        backend_status = "healthy"
    elif cpu < 95 and ram < 95 and disk < 98:
        backend_status = "degraded"
    else:
        backend_status = "critical"

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO system_status (timestamp, backend, cpu_usage, ram_usage) VALUES (%s, %s, %s, %s)",
            (datetime.now(), backend_status, cpu, ram)
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("⚠ Failed to save system history:", e)

    return JSONResponse({
        "backend": backend_status,
        "database": db_status,
        "cpu_usage": f"{cpu}%",
        "ram_usage": f"{ram}%",
        "disk_usage": f"{disk}%",
        "model_status": "ready",
        "server": "Apache / Gunicorn",
        "technologies": {
            "frontend": "ReactJS",
            "backend": "FastAPI",
            "ml_model": "Facebook Prophet",
            "database": "MySQL",
            "deployment": "Gunicorn",
            "os": "AlmaLinux 9"
        },
        "timestamp": datetime.now().isoformat()
    })


# ============================================================
# STATUS HISTORY
# ============================================================
@router.get("/status/history")
def status_history():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT timestamp, backend, cpu_usage, ram_usage
            FROM system_status
            WHERE timestamp >= NOW() - INTERVAL 24 HOUR
            ORDER BY timestamp ASC
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return {"history": rows}
    except Exception as e:
        return {"history": [], "error": str(e)}


# ============================================================
# RESTART SERVICE
# ============================================================
from fastapi import Header, HTTPException
import os

@router.post("/status/restart")
def restart_backend(admin_key: str = Header(None)):
    if admin_key != "AirQ-Admin-2025":
        raise HTTPException(status_code=403, detail="Unauthorized")
    os.system("systemctl restart fastapi-airq")
    return {"message": "Backend restarted successfully"}
