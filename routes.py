import io
import os
import time
import json
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

import pandas as pd
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Request,
    Header,
    HTTPException,
)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from db import get_db_connection
from helpers import fetch_all_data, get_data_info, detect_outliers
from ml import (
    process_basic_forecast,
    process_advanced_forecast_stream,
)

router = APIRouter(
    prefix="/api/v1",
    tags=["Air Quality Data & Forecasting"],
)

# ============================================================
# Pydantic Models (for API docs & validation)
# ============================================================

class AirQualityRecord(BaseModel):
    """Single air quality measurement row."""
    model_config = ConfigDict(extra="ignore")

    id: Optional[int] = None
    waktu: datetime
    pm10: Optional[float] = None
    pm25: Optional[float] = None
    so2: Optional[float] = None
    co: Optional[float] = None
    o3: Optional[float] = None
    no2: Optional[float] = None
    hc: Optional[float] = None
    kelembaban: Optional[float] = None
    suhu: Optional[float] = None


class GetAllDataResponse(BaseModel):
    """Response for /data endpoint."""
    data: List[AirQualityRecord]
    status: Literal["ok"]


class DataInfoResponse(BaseModel):
    """Summary information about dataset quality."""
    totalData: int
    outlierClear: bool
    nanClear: bool
    outlierCount: int
    nanCount: int
    message: Optional[str] = None


class OutliersResponse(BaseModel):
    """Detected outlier rows."""
    outliers: List[Dict[str, Any]]
    status: Literal["ok"]


class MessageResponse(BaseModel):
    """Simple message-only response."""
    message: str


class ForecastResponse(BaseModel):
    """Basic forecast response wrapper."""
    message: str
    forecast: Dict[str, Any]


class ActivityLogEntry(BaseModel):
    """Single activity log entry."""
    model_config = ConfigDict(extra="ignore")

    id: Optional[int] = None
    timestamp: Optional[datetime] = None
    event: str
    detail: Optional[str] = None


class ActivityLogListResponse(BaseModel):
    """Response for activity log list."""
    log: List[ActivityLogEntry]


class ActivityLogCreateRequest(BaseModel):
    """Request body for creating a new activity log."""
    event: str
    detail: Optional[str] = None


class StatusHistoryEntry(BaseModel):
    """Single status history point."""
    model_config = ConfigDict(extra="ignore")

    timestamp: datetime
    backend: str
    cpu_usage: float
    ram_usage: float


class StatusHistoryResponse(BaseModel):
    """Response for status history endpoint."""
    history: List[StatusHistoryEntry]
    error: Optional[str] = None


class SystemStatusResponse(BaseModel):
    """Current backend, database & resource status."""
    backend: str
    database: str
    cpu_usage: str
    ram_usage: str
    disk_usage: str
    model_status: str
    server: str
    technologies: Dict[str, str]
    timestamp: str


# ============================================================
# 1. GET All Air Quality Data
# ============================================================

@router.get(
    "/data",
    response_model=GetAllDataResponse,
    summary="Get all air quality data",
    response_description="Returns all recorded air quality measurements."
)
def get_all_data():
    """
    Fetch all air quality measurement records from the database.

    This endpoint is used by the frontend to populate the main data table.
    """
    try:
        rows = fetch_all_data()
        # Biarkan Pydantic + FastAPI yang handle konversi datetime
        records = [AirQualityRecord(**dict(r)) for r in rows]
        return GetAllDataResponse(data=records, status="ok")
    except Exception as e:
        traceback.print_exc()
        # Untuk docs, biarkan FastAPI yang wrap error jadi JSON
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 2. DATA INFO
# ============================================================

@router.get(
    "/data/info",
    response_model=DataInfoResponse,
    summary="Get dataset info & data quality",
    response_description="Returns count, NaN & outlier summary for dataset."
)
def get_info():
    """
    Return information about the dataset quality:
    - Total rows
    - Number of NaNs
    - Number of outliers
    - Flags for 'clean' status
    """
    try:
        rows = fetch_all_data()
        if not rows:
            # default structure when no data available
            return DataInfoResponse(
                totalData=0,
                outlierClear=True,
                nanClear=True,
                outlierCount=0,
                nanCount=0,
                message="No data available yet.",
            )

        info = get_data_info(rows)
        return DataInfoResponse(**info)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 3. GET Outliers
# ============================================================

@router.get(
    "/data/outliers",
    response_model=OutliersResponse,
    summary="Get detected outliers",
    response_description="Returns list of detected outliers for all pollutants."
)
def get_outliers():
    """
    Detect outliers in the full dataset and return them to the frontend.

    Detection logic is handled inside `detect_outliers(df)`.
    """
    try:
        rows = fetch_all_data()
        if not rows:
            return OutliersResponse(outliers=[], status="ok")

        df = pd.DataFrame(rows)
        outliers = detect_outliers(df)
        return OutliersResponse(outliers=outliers, status="ok")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 4. Handle Outliers (Interpolate)
# ============================================================

@router.post(
    "/data/outliers-handle",
    response_model=MessageResponse,
    summary="Handle outliers by interpolation",
    response_description="Replaces outlier values using linear interpolation."
)
def handle_outliers():
    """
    Detect outliers (3-sigma rule) in numeric columns and replace them
    using linear interpolation along the time axis.

    Affected table: `air_quality_data`.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM air_quality_data ORDER BY waktu ASC")
        rows = cursor.fetchall()

        if not rows:
            cursor.close()
            conn.close()
            return MessageResponse(message="No data available.")

        df = pd.DataFrame(rows)
        df["waktu"] = pd.to_datetime(df["waktu"])
        numeric_cols = [
            "pm10", "pm25", "so2", "co",
            "o3", "no2", "hc",
            "kelembaban", "suhu",
        ]

        outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)
        for col in numeric_cols:
            mean, std = df[col].mean(), df[col].std()
            outlier_mask[col] = (df[col] - mean).abs() > 3 * std

        if not outlier_mask.values.any():
            cursor.close()
            conn.close()
            return MessageResponse(message="No outliers detected.")

        # Replace outliers with NaN then interpolate
        for col in numeric_cols:
            df.loc[outlier_mask[col], col] = None
            df[col] = df[col].interpolate(method="linear", limit_direction="both")

        # Persist back to DB
        for _, row in df.iterrows():
            sql = """
                UPDATE air_quality_data
                SET pm10=%s, pm25=%s, so2=%s, co=%s, o3=%s, no2=%s, hc=%s,
                    kelembaban=%s, suhu=%s
                WHERE id=%s
            """
            cursor.execute(
                sql,
                (
                    row["pm10"], row["pm25"], row["so2"], row["co"],
                    row["o3"], row["no2"], row["hc"],
                    row["kelembaban"], row["suhu"],
                    row["id"],
                ),
            )

        conn.commit()
        cursor.close()
        conn.close()

        return MessageResponse(
            message=f"{int(outlier_mask.values.sum())} outlier values interpolated successfully."
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 5. Delete All Data
# ============================================================

@router.delete(
    "/data/delete-all",
    response_model=MessageResponse,
    summary="Delete all air quality data",
    response_description="Deletes all rows from `air_quality_data` table."
)
def delete_all_data():
    """
    Hard-delete all records from `air_quality_data`.
    Use with caution.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM air_quality_data")
        conn.commit()
        cursor.close()
        conn.close()
        return MessageResponse(
            message="All data records have been deleted successfully."
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 6. Upload CSV
# ============================================================
def clean_date_to_dateonly(dt: datetime):
    try:
        return dt.date()   # hanya tanggal, tanpa jam
    except:
        return dt

@router.post(
    "/upload-csv",
    response_model=MessageResponse,
    summary="Upload CSV dataset",
    response_description="Imports CSV file into `air_quality_data` table."
)
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file and import it into `air_quality_data`.

    Required columns:
    - waktu, pm10, pm25, so2, co, o3, no2, hc

    Optional columns:
    - kelembaban, suhu
    """
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail="File must be in CSV format.",
            )

        content = await file.read()
        data = pd.read_csv(io.StringIO(content.decode("utf-8")))

        required_cols = ["waktu", "pm10", "pm25", "so2", "co", "o3", "no2", "hc"]
        for col in required_cols:
            if col not in data.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing column '{col}'",
                )

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
                    clean_date_to_dateonly(r["waktu"]).strftime("%Y-%m-%d"),
                    r["pm10"], r["pm25"], r["so2"], r["co"],
                    r["o3"], r["no2"], r["hc"],
                    r["kelembaban"], r["suhu"],
                ),
            )

        conn.commit()
        cursor.close()
        conn.close()
        return MessageResponse(message="CSV file uploaded and saved successfully.")
    except HTTPException:
        # biarkan HTTPException bubble ke FastAPI
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 7. Manual Data Input
# ============================================================

@router.post(
    "/input",
    response_model=MessageResponse,
    summary="Insert single air quality record",
    response_description="Inserts one row into `air_quality_data` from JSON body."
)
async def input_air_quality(request: Request):
    """
    Manually insert a single air quality record.

    Body fields expected:
    - waktu, pm10, pm25, so2, co, o3, no2, hc, kelembaban, suhu
    """
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
                data.get("waktu"),
                data.get("pm10"),
                data.get("pm25"),
                data.get("so2"),
                data.get("co"),
                data.get("o3"),
                data.get("no2"),
                data.get("hc"),
                data.get("kelembaban"),
                data.get("suhu"),
            ),
        )
        conn.commit()
        cursor.close()
        conn.close()
        return MessageResponse(
            message="Record inserted successfully into database."
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 8. Process Forecast Basic
# ============================================================

@router.post(
    "/model/process-basic",
    response_model=ForecastResponse,
    summary="Run basic Prophet forecast for all pollutants",
    response_description="Returns forecast results for all pollutants."
)
def process_basic_all():
    """
    Run a non-streaming Prophet forecast for all pollutants
    (pm10, pm25, so2, o3, no2, co, hc) and return the result.
    """
    try:
        rows = fetch_all_data()
        if not rows:
            raise HTTPException(
                status_code=400,
                detail="No data available for processing.",
            )

        df = pd.DataFrame(rows)
        df["waktu"] = pd.to_datetime(df["waktu"])
        pollutants = ["pm10", "pm25", "so2", "o3", "no2", "co", "hc"]
        forecast = process_basic_forecast(df, pollutants)

        return ForecastResponse(
            message="Forecast Prophet successfully processed for all pollutants.",
            forecast=forecast,
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 9. Process Forecast Advanced (Streaming)
# ============================================================

@router.post(
    "/model/process-advanced",
    summary="Run advanced (streaming) Prophet forecast",
    response_description="Server-Sent Events (SSE) stream with progress updates.",
)
def process_advanced_all():
    """
    Run advanced Prophet forecast with progress updates via Server-Sent Events.

    Frontend should consume this as an `EventSource` stream.
    """
    def progress_stream():
        try:
            rows = fetch_all_data()
            if not rows:
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "status": "error",
                            "message": "No data available for processing.",
                        }
                    )
                    + "\n\n"
                )
                return

            df = pd.DataFrame(rows)
            df["waktu"] = pd.to_datetime(df["waktu"])
            pollutants = ["pm10", "pm25", "so2", "o3", "no2", "co", "hc"]
            total = len(pollutants)

            yield (
                "data: "
                + json.dumps(
                    {
                        "status": "start",
                        "total": total,
                        "message": "Starting advanced forecast...",
                    }
                )
                + "\n\n"
            )

            for idx, pol in enumerate(pollutants, start=1):
                try:
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "status": "processing",
                                "pollutant": pol.upper(),
                                "progress": round((idx - 1) / total * 100, 2),
                            }
                        )
                        + "\n\n"
                    )
                    process_advanced_forecast_stream(df, [pol])
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "status": "done",
                                "pollutant": pol.upper(),
                                "progress": round(idx / total * 100, 2),
                            }
                        )
                        + "\n\n"
                    )
                except Exception as e:
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "status": "error",
                                "pollutant": pol.upper(),
                                "message": str(e),
                            }
                        )
                        + "\n\n"
                    )
                    continue

                time.sleep(0.3)

            yield (
                "data: "
                + json.dumps(
                    {
                        "status": "complete",
                        "progress": 100,
                        "message": "All forecasts processed successfully!",
                    }
                )
                + "\n\n"
            )
        except Exception as e:
            traceback.print_exc()
            yield (
                "data: "
                + json.dumps({"status": "error", "message": str(e)})
                + "\n\n"
            )

    return StreamingResponse(progress_stream(), media_type="text/event-stream")

@router.get(
    "/forecast/{pol}/advanced",
    summary="Get advanced forecast result for specific pollutant",
)
def get_advanced_forecast_by_pollutant(pol: str):
    """
    Return advanced forecast data for the given pollutant.
    Table: forecast_{pol}_with_parameters_data
    """
    allowed = ["pm10", "pm25", "so2", "co", "o3", "no2", "hc"]
    pol = pol.lower()

    if pol not in allowed:
        raise HTTPException(status_code=400, detail="Invalid pollutant name")

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        table = f"forecast_{pol}_with_parameters_data"
        cursor.execute(f"SELECT * FROM {table} ORDER BY ds ASC")
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        return rows

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 10. Clear Forecast Tables
# ============================================================

@router.delete(
    "/model/clear-forecast",
    response_model=MessageResponse,
    summary="Clear all forecast tables",
    response_description="TRUNCATE all forecast_* tables."
)
def clear_forecast():
    """
    Truncate all forecast-related tables in the database.
    """
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
        return MessageResponse(
            message="All forecast tables have been cleared successfully."
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 11. SYSTEM STATUS & TECHNOLOGIES
# ============================================================

@router.get(
    "/status",
    response_model=SystemStatusResponse,
    summary="Get system status",
    response_description="Returns backend, DB and resource usage status."
)
def system_status():
    """
    Get current system status including:
    - backend health
    - database connection status
    - CPU, RAM, disk usage
    - technologies information

    Also persists a snapshot into `system_status` table.
    """
    import psutil
    import shutil

    try:
        conn = get_db_connection()
        if conn:
            conn.cursor().execute("SELECT 1")
            conn.close()
            db_status = "connected"
        else:
            db_status = "disconnected"
    except Exception:
        db_status = "disconnected"

    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory().percent
    disk_info = shutil.disk_usage("/")
    # Persentase disk terpakai
    disk = round(disk_info.used / disk_info.total * 100, 2)

    if cpu < 80 and ram < 80 and disk < 90:
        backend_status = "healthy"
    elif cpu < 95 and ram < 95 and disk < 98:
        backend_status = "degraded"
    else:
        backend_status = "critical"

    # Simpan history status ke DB (best-effort)
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO system_status (timestamp, backend, cpu_usage, ram_usage)
            VALUES (%s, %s, %s, %s)
            """,
            (datetime.now(), backend_status, cpu, ram),
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("⚠ Failed to save system history:", e)

    return SystemStatusResponse(
        backend=backend_status,
        database=db_status,
        cpu_usage=f"{cpu}%",
        ram_usage=f"{ram}%",
        disk_usage=f"{disk}%",
        model_status="ready",
        server="Apache / Gunicorn",
        technologies={
            "frontend": "ReactJS",
            "backend": "FastAPI",
            "ml_model": "Facebook Prophet",
            "database": "MySQL",
            "deployment": "Gunicorn",
            "os": "AlmaLinux 9",
        },
        timestamp=datetime.now().isoformat(),
    )


# ============================================================
# 12. Activity Log
# ============================================================

@router.get(
    "/activity-log",
    response_model=ActivityLogListResponse,
    summary="Get activity log list",
    response_description="Returns the last 200 model activity logs."
)
def get_activity_log():
    """
    Return the latest 200 activity log entries, ordered by timestamp DESC.

    Used by the ModelPage frontend to show recent actions.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM activity_log ORDER BY timestamp DESC LIMIT 200"
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        logs = [ActivityLogEntry(**row) for row in rows]
        return ActivityLogListResponse(log=logs)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/activity-log/add",
    response_model=MessageResponse,
    summary="Add new activity log entry",
    response_description="Inserts a new activity log row into database."
)
async def add_activity_log(payload: ActivityLogCreateRequest):
    """
    Insert a new activity log record.

    This endpoint is intended to be called from the frontend ModelPage.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO activity_log (event, detail) VALUES (%s, %s)",
            (payload.event, payload.detail),
        )
        conn.commit()
        cursor.close()
        conn.close()
        return MessageResponse(message="Activity logged")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 13. STATUS HISTORY
# ============================================================

@router.get(
    "/status/history",
    response_model=StatusHistoryResponse,
    summary="Get last 24 hours backend status history",
    response_description="Returns history points from `system_status` table."
)
def status_history():
    """
    Return system status history for the last 24 hours
    from the `system_status` table.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT timestamp, backend, cpu_usage, ram_usage
            FROM system_status
            WHERE timestamp >= NOW() - INTERVAL 24 HOUR
            ORDER BY timestamp ASC
            """
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        history = [StatusHistoryEntry(**row) for row in rows]
        return StatusHistoryResponse(history=history)
    except Exception as e:
        return StatusHistoryResponse(history=[], error=str(e))


# ============================================================
# 14. RESTART SERVICE
# ============================================================

@router.post(
    "/status/restart",
    response_model=MessageResponse,
    summary="Restart FastAPI backend service",
    response_description="Restarts systemd service `fastapi-airq` (admin only).",
)
def restart_backend(admin_key: str = Header(None, description="Admin secret key")):
    """
    Restart the FastAPI backend service via `systemctl restart fastapi-airq`.

    This endpoint is protected by a simple header-based key.
    """
    if admin_key != "AirQ-Admin-2025":
        raise HTTPException(status_code=403, detail="Unauthorized")

    os.system("systemctl restart fastapi-airq")
    return MessageResponse(message="Backend restarted successfully")

@router.get("/forecast/check-basic")
def check_basic_forecast():
    """
    Check whether basic forecast data already exists
    on any pollutant forecast table.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        tables = [
            "forecast_pm10_data",
            "forecast_pm25_data",
            "forecast_so2_data",
            "forecast_co_data",
            "forecast_o3_data",
            "forecast_no2_data",
            "forecast_hc_data",
        ]

        exists = False
        for t in tables:
            cursor.execute(f"SELECT COUNT(*) AS c FROM {t}")
            row = cursor.fetchone()
            if row["c"] > 0:
                exists = True
                break

        cursor.close()
        conn.close()

        return {"exists": exists}

    except Exception as e:
        return {"exists": False, "error": str(e)}

@router.get("/forecast/check-advanced")
def check_advanced_forecast():
    """
    Check whether advanced forecast data already exists
    on any forecast_{pol}_with_parameters_data table.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        tables = [
            "forecast_pm10_with_parameters_data",
            "forecast_pm25_with_parameters_data",
            "forecast_so2_with_parameters_data",
            "forecast_co_with_parameters_data",
            "forecast_o3_with_parameters_data",
            "forecast_no2_with_parameters_data",
            "forecast_hc_with_parameters_data",
        ]

        exists = False
        for t in tables:
            cursor.execute(f"SELECT COUNT(*) AS c FROM {t}")
            row = cursor.fetchone()
            if row["c"] > 0:
                exists = True
                break

        cursor.close()
        conn.close()

        return {"exists": exists}

    except Exception as e:
        return {"exists": False, "error": str(e)}

@router.get(
    "/forecast/{pol}",
    summary="Get forecast result for specific pollutant",
)
def get_forecast_by_pollutant(pol: str):
    """
    Return forecast data for the given pollutant.
    Table: forecast_{pol}_data
    """
    allowed = ["pm10", "pm25", "so2", "co", "o3", "no2", "hc"]
    pol = pol.lower()

    if pol not in allowed:
        raise HTTPException(status_code=400, detail="Invalid pollutant name")

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        table = f"forecast_{pol}_data"
        cursor.execute(f"SELECT * FROM {table} ORDER BY ds ASC")
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
