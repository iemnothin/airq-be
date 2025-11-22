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

@router.get(
    "/data",
    summary="[AirQ] Retrieve all air quality data",
    description="""
    Fetch all recorded air quality data from the `air_quality_data` table.

    Each record represents a single timestamp (e.g., hourly) with pollutant concentrations and meteorological parameters.

    ---
    ### 🔹 Example cURL:
    ```bash
    curl -X GET "https://airq.abiila.com/api/v1/data" -H "accept: application/json"
    ```

    ### 🔹 Example Response:
    ```json
    {
      "status": "ok",
      "data": [
        {
          "id": 1,
          "waktu": "2025-11-12 00:00:00",
          "pm10": 42.1,
          "pm25": 17.8,
          "so2": 4.2,
          "co": 341,
          "o3": 32,
          "no2": 15,
          "hc": 371,
          "kelembaban": 72,
          "suhu": 29
        }
      ]
    }
    ```
    """,
    responses={
        200: {"description": "Data retrieved successfully"},
        500: {"description": "Internal server error"},
    },
)
def get_all_data():
    """
    Retrieve all historical air quality records from the database.

    Used by dashboard and analytics modules to display pollutant trends.
    """
    try:
        rows = fetch_all_data()
        return JSONResponse(content={"data": rows, "status": "ok"}, status_code=200)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ============================================================
# 2. DATA INFO (for dashboard cards)
# ============================================================
@router.get(
    "/data/info",
    summary="[AirQ] Get dataset summary",
    description="Returns dataset summary or default values if no data is available.",
    responses={
        200: {"description": "Dataset summary returned or default values when empty"},
        500: {"description": "Internal server error"},
    },
)
def get_info():
    """
    Returns summary information used by dashboard cards:
    - totalData: total number of rows
    - outlierClear / nanClear: boolean indicators
    - outlierCount / nanCount: counts of problematic values

    If dataset is empty, returns sensible defaults so the dashboard remains usable.
    """
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
@router.get(
    "/data/outliers",
    summary="[AirQ] Detect outliers in dataset",
    description="Detect outliers safely even when dataset is empty.",
    responses={
        200: {"description": "Outliers detected and returned"},
        500: {"description": "Internal server error"},
    },
)
def get_outliers():
    """
    Detects statistical outliers in numeric pollutant columns.

    Returns a list of outlier records or an empty list when no data is available.
    """
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
@router.post(
    "/data/outliers-handle",
    summary="[AirQ] Interpolate detected outliers",
    description="""
    Automatically replaces detected outlier values in numeric columns using **linear interpolation**.

    Process:
    1. Detects outliers where |x - μ| > 3σ
    2. Sets those values to `null`
    3. Fills gaps using linear interpolation across time (`waktu`)

    Updates the database in place.

    **Response example:**
    ```json
    {"message": "12 outlier values have been interpolated successfully."}
    ```
    """,
    responses={
        200: {"description": "Outliers interpolated and DB updated"},
        200: {"description": "No outliers detected (no changes)"},
        500: {"description": "Internal server error"},
    },
)
def handle_outliers():
    """
    Finds outliers in numeric pollutant columns and replaces them by linear interpolation.

    Numeric columns processed: pm10, pm25, so2, co, o3, no2, hc, kelembaban, suhu.

    The function updates the `air_quality_data` table row-by-row with interpolated values.
    """
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
        numeric_cols = [
            "pm10",
            "pm25",
            "so2",
            "co",
            "o3",
            "no2",
            "hc",
            "kelembaban",
            "suhu",
        ]

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
                (
                    row["pm10"],
                    row["pm25"],
                    row["so2"],
                    row["co"],
                    row["o3"],
                    row["no2"],
                    row["hc"],
                    row["kelembaban"],
                    row["suhu"],
                    row["id"],
                ),
            )
        conn.commit()
        cursor.close()
        conn.close()

        return JSONResponse(
            content={
                "message": f"{int(outlier_mask.values.sum())} outlier values interpolated successfully."
            },
            status_code=200,
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ============================================================
# 5. Delete All Data
# ============================================================
@router.delete(
    "/data/delete-all",
    summary="[AirQ] Delete all air quality data",
    description="""
    Permanently removes **all records** from the `air_quality_data` table.

    ⚠️ **Warning:** This action cannot be undone.
    Use with caution in administrative or reset scenarios.

    ---
    ### 🔹 Example cURL:
    ```bash
    curl -X DELETE "https://airq.abiila.com/api/v1/data/delete-all" -H "accept: application/json"
    ```

    ### 🔹 Example Response:
    ```json
    {"message": "All data records have been deleted successfully."}
    ```
    """,
    responses={
        200: {"description": "All records deleted successfully"},
        500: {"description": "Internal server error"},
    },
)
def delete_all_data():
    """
    Deletes every row from `air_quality_data`. Intended for administrative resets.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM air_quality_data")
        conn.commit()
        cursor.close()
        conn.close()
        return JSONResponse(content={"message": "All data records have been deleted successfully."}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ============================================================
# 6. Upload CSV
# ============================================================
@router.post(
    "/upload-csv",
    summary="[AirQ] Upload CSV dataset",
    description="""
    Upload a CSV file containing historical air quality measurements and insert it into the database.

    Each record represents one timestamp (`Waktu`) and pollutant concentration values.

    ---
    ### 🔹 Required Columns:
    - `Waktu`, `PM10`, `PM25`, `SO2`, `CO`, `O3`, `NO2`, `HC`
    ### 🔹 Optional Columns:
    - `Kelembaban`, `Suhu`

    ---
    ### 🔹 Example cURL:
    ```bash
    curl -X POST "https://airq.abiila.com/api/v1/upload-csv" \
         -H "accept: application/json" \
         -F "file=@dataset.csv"
    ```

    ### 🔹 Example Response:
    ```json
    {"message": "CSV file uploaded and saved successfully."}
    ```
    """,
    responses={
        200: {"description": "CSV file uploaded successfully"},
        400: {"description": "Invalid CSV file or missing columns"},
        500: {"description": "Internal server error"},
    },
)
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload CSV dataset and import into `air_quality_data` table.

    Converts `Waktu` into datetime and skips rows with invalid timestamps.
    """
    try:
        if not file.filename.endswith(".csv"):
            return JSONResponse({"error": "File must be in CSV format."}, status_code=400)

        content = await file.read()
        data = pd.read_csv(io.StringIO(content.decode("utf-8")))

        required_cols = [
            "waktu",
            "pm10",
            "pm25",
            "so2",
            "co",
            "o3",
            "no2",
            "hc",
        ]
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
        insert_sql = """
            INSERT INTO air_quality_data
            (waktu, pm10, pm25, so2, co, o3, no2, hc, kelembaban, suhu)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """

        for _, r in data.iterrows():
            cursor.execute(
                insert_sql,
                (
                    r["waktu"].strftime("%Y-%m-%d %H:%M:%S"),
                    r["pm10"],
                    r["pm25"],
                    r["so2"],
                    r["co"],
                    r["o3"],
                    r["no2"],
                    r["hc"],
                    r["kelembaban"],
                    r["suhu"],
                ),
            )

        conn.commit()
        cursor.close()
        conn.close()
        return JSONResponse(content={"message": "CSV file uploaded and saved successfully."}, status_code=200)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# 7. Manual Data Input
# ============================================================
@router.post(
    "/input",
    summary="[AirQ] Manually insert new record",
    description="Insert a single record manually into the air quality dataset via JSON body.",
    responses={
        200: {"description": "Reco  rd inserted successfully"},
        500: {"description": "Internal server error"},
    },
)
async def input_air_quality(request: Request):
    """
    Insert a single measurement record into `air_quality_data`.

    Example request body:
    ```json
    {
        "waktu": "2024-07-30 00:00:00",
        "pm10": 74,
        "pm25": 15,
        "so2": 6,
        "co": 351,
        "o3": 33,
        "no2": 17,
        "hc": 388,
        "kelembaban": 72,
        "suhu": 30
    }
    ```
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
        return JSONResponse(content={"message": "Record inserted successfully into database."}, status_code=200)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ============================================================
# 8. Process Forecast Basic
# ============================================================
@router.post(
    "/model/process-basic",
    summary="[AirQ] Generate Prophet forecast (Basic Model)",
    description="""
    Trains a **basic Prophet model** for all 7 pollutants (`PM10`, `PM25`, `SO2`, `O3`, `NO2`, `CO`, `HC`).

    - Uses default Prophet settings (yearly + weekly seasonality)
    - Produces 30-day future forecast for each pollutant
    - Saves results into respective tables:
      - `forecast_pm10_data`, `forecast_pm25_data`, etc.

    **Response example:**
    ```json
    {
      "message": "Forecast Prophet successfully processed for all pollutants.",
      "forecast": {
        "pm10": [
          {"ds": "2025-11-13", "yhat": 42.3, "yhat_lower": 37.8, "yhat_upper": 48.2},
          {"ds": "2025-11-14", "yhat": 43.1, "yhat_lower": 38.1, "yhat_upper": 49.0}
        ],
        "pm25": [
          {"ds": "2025-11-13", "yhat": 18.4, "yhat_lower": 15.1, "yhat_upper": 22.6}
        ]
      }
    }
    ```

    ---
    ### 🔹 Example cURL:
    ```bash
    curl -X POST "https://airq.abiila.com/api/v1/model/process-basic" -H "accept: application/json"
    ```
    """,
    responses={
        200: {"description": "Forecast successfully processed"},
        400: {"description": "No data available for processing"},
        500: {"description": "Internal server error"},
    },
)
def process_basic_all():
    """
    Run a Prophet forecast with default parameters for all pollutants.
    Returns prediction data ready for visualization and evaluation.
    """
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
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ============================================================
# 9. Process Forecast Advanced
# ============================================================
@router.post(
    "/model/process-advanced",
    summary="[AirQ] Generate optimized forecast (Advanced Model)",
    description="""
    Trains an **advanced Prophet model** for all 7 pollutants (`PM10`, `PM25`, `SO2`, `O3`, `NO2`, `CO`, `HC`).

    Features:
    - Uses parameter optimization (changepoint, seasonality, holiday priors)
    - Includes **Indonesian public holidays** as regressors
    - Performs cross-validation to choose best hyperparameters
    - Produces 30-day future forecast
    - Saves results into tables:
      - `forecast_pm10_with_parameters_data`, `forecast_pm25_with_parameters_data`, etc.

    **Response example:**
    ```json
    {
      "message": "Forecast with parameters (7 pollutants) successfully processed.",
      "forecast": {
        "pm10": [
          {"ds": "2025-11-13", "yhat": 41.8, "yhat_lower": 36.2, "yhat_upper": 47.5}
        ]
      }
    }
    ```

    ---
    ### 🔹 Example cURL:
    ```bash
    curl -X POST "https://airq.abiila.com/api/v1/model/process-advanced" -H "accept: application/json"
    ```
    """,
    responses={
        200: {"description": "Advanced forecast processed"},
        400: {"description": "No data available for processing"},
        500: {"description": "Internal server error"},
    },
)
def process_advanced_all():
    """
    Run an advanced Prophet workflow with hyperparameter optimization and holidays.

    This endpoint streams progress internally but returns when processing completes.
    """
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

            yield f"data: {json.dumps({'status': 'start', 'message': 'Starting advanced forecast...', 'total': total})}\n\n"

            for idx, pol in enumerate(pollutants, start=1):
                try:
                    yield f"data: {json.dumps({'status': 'processing', 'pollutant': pol.upper(), 'progress': round((idx - 1) / total * 100, 2)})}\n\n"
                    result = process_advanced_forecast_stream(df, [pol])
                    yield f"data: {json.dumps({'status': 'done', 'pollutant': pol.upper(), 'progress': round(idx / total * 100, 2)})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'status': 'error', 'pollutant': pol.upper(), 'message': str(e)})}\n\n"
                    continue

                # optional delay for smoother progress bar updates
                time.sleep(0.3)

            yield f"data: {json.dumps({'status': 'complete', 'progress': 100, 'message': 'All forecasts processed successfully!'})}\n\n"

        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(progress_stream(), media_type="text/event-stream")


# ============================================================
# 9B. Process Forecast Advanced (Streaming Progress)
# ============================================================
@router.get(
    "/model/process-advanced/stream",
    summary="[AirQ] Stream advanced forecast progress (real-time)",
    description="""
    Streams live progress updates for advanced forecast processing (per pollutant).
    Uses **Server-Sent Events (SSE)**, allowing the frontend to show real-time status and progress bar.

    Example stream output:
    ```
    data: {"status":"processing","pollutant":"PM10","progress":14.2}
    data: {"status":"done","pollutant":"PM25","progress":100}
    data: {"status":"complete","progress":100,"message":"All forecasts done!"}
    ```
    """,
    responses={
        200: {"description": "SSE stream of progress events"},
        500: {"description": "Internal server error"},
    },
)
def process_advanced_stream():
    """
    SSE endpoint for real-time progress of the advanced Prophet forecast.

    The frontend should connect using EventSource to receive `data:` events.
    """
    def event_stream():
        rows = fetch_all_data()
        if not rows:
            yield f"data: {json.dumps({'status': 'error', 'message': 'No data available for forecasting'})}\n\n"
            return

        df = pd.DataFrame(rows)
        df["waktu"] = pd.to_datetime(df["waktu"])
        pollutants = ["pm10", "pm25", "so2", "o3", "no2", "co", "hc"]

        yield f"data: {json.dumps({'status': 'start', 'message': 'Starting advanced forecasting for all pollutants'})}\n\n"

        for event in process_advanced_forecast_stream(df, pollutants):
            yield event

        yield f"data: {json.dumps({'status': 'complete', 'progress': 100, 'message': '✅ All forecasts completed successfully'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ============================================================
# 10. Clear Forecast Tables
# ============================================================
@router.delete(
    "/model/clear-forecast",
    summary="[AirQ] Clear all forecast result tables",
    description="""
    Truncates (clears) all forecast result tables in the database, including both basic and parameterized model outputs:

    - `forecast_pm10_data`, `forecast_pm10_with_parameters_data`
    - `forecast_pm25_data`, `forecast_pm25_with_parameters_data`
    - ... and others (`SO2`, `O3`, `NO2`, `CO`, `HC`)

    **Response example:**
    ```json
    {"message": "All forecast tables have been cleared successfully."}
    ```
    """,
    responses={
        200: {"description": "All forecast tables cleared successfully"},
        500: {"description": "Internal server error"},
    },
)
def clear_forecast():
    """
    Truncates all forecast result tables used by the AirQ forecasting pipelines.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        tables = [
            "forecast_pm10_data",
            "forecast_pm10_with_parameters_data",
            "forecast_pm25_data",
            "forecast_pm25_with_parameters_data",
            "forecast_so2_data",
            "forecast_so2_with_parameters_data",
            "forecast_o3_data",
            "forecast_o3_with_parameters_data",
            "forecast_no2_data",
            "forecast_no2_with_parameters_data",
            "forecast_co_data",
            "forecast_co_with_parameters_data",
            "forecast_hc_data",
            "forecast_hc_with_parameters_data",
        ]
        for t in tables:
            cursor.execute(f"TRUNCATE TABLE {t}")
        conn.commit()
        cursor.close()
        conn.close()
        return JSONResponse(content={"message": "All forecast tables have been cleared successfully."}, status_code=200)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# 11. SYSTEM STATUS & TECHNOLOGIES
# ============================================================
@router.get(
    "/status",
    summary="[AirQ] Check backend system and technology status",
    description="""
    Returns the current operational status of the **AirQ system**, including backend, database, and model readiness.
    Also lists all technologies used in this project stack.

    **Response Example:**
    ```json
    {
      "backend": "online",
      "database": "connected",
      "model_status": "ready",
      "server": "Apache",
      "technologies": {
        "frontend": "ReactJS",
        "backend": "FastAPI",
        "ml_model": "Facebook Prophet",
 "database": "MySQL",
        "deployment": "Gunicorn",
        "os": "AlmaLinux 9"
      },
      "timestamp": "2025-11-09T20:45:10"
    }
    ```
    """,
    responses={
        200: {"description": "System status returned"},
        500: {"description": "Internal server error"},
    },
)
def system_status():
    """
    Checks database connectivity and returns a summary of services and technologies.
    """
    import psutil
    import shutil
    from datetime import datetime
    from fastapi.responses import JSONResponse
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
    disk = shutil.disk_usage("/").percent if hasattr(shutil.disk_usage("/"), 'percent') else 0

    if cpu < 80 and ram < 80 and disk < 90:
        backend_status = "healthy"
    elif cpu < 95 and ram < 95 and disk < 98:
        backend_status = "degraded"
    else:
        backend_status = "critical"

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