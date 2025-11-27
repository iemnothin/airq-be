from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.make_holidays import make_holidays_df
from prophet.serialize import model_to_json, model_from_json
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import itertools, json
from datetime import datetime
from db import get_db_connection

model_cache = {}

def get_or_train_model(df, column, years=[2022,2023,2024,2025,2026]):
    if column not in model_cache:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=make_holidays_df(year_list=years, country="ID"),
        )
        model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        data = df[["Waktu", column]].rename(columns={"Waktu":"ds", column:"y"})
        model.fit(data)
        model_cache[column] = model
    return model_cache[column]

def process_basic_forecast(df_full, pollutants):
    results = {}
    for pol in pollutants:
        df = df_full[["waktu", pol]].rename(columns={"waktu": "ds", pol: "y"}).dropna(subset=["y"])
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        model.fit(df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        result = forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(30)
        result["ds"] = result["ds"].dt.strftime("%Y-%m-%d")
        results[pol] = result.round(2).to_dict(orient="records")

        # simpan ke DB
        table_name = f"forecast_{pol}_data"
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {table_name}")
        insert = f"INSERT INTO {table_name} (ds,yhat,yhat_lower,yhat_upper) VALUES (%s,%s,%s,%s)"
        for _, row in result.iterrows():
            cursor.execute(insert, (
                datetime.strptime(row["ds"], "%Y-%m-%d").date(),
                float(row["yhat"]), float(row["yhat_lower"]), float(row["yhat_upper"])
            ))
        conn.commit(); cursor.close(); conn.close()
    return results

def process_advanced_forecast_stream(df_full, pollutants):
    """
    Streaming version of advanced forecast — yields progress updates as strings.
    """
    holidays = make_holidays_df(year_list=[2022, 2023, 2024, 2025, 2026], country="ID")
    param_grid = list(itertools.product(
        [0.05, 0.1, 0.2],
        [1.0, 5.0, 10.0],
        [1.0, 5.0, 10.0],
        [True, False],
        [True, False]
    ))

    for pol in pollutants:
        yield f"data: {json.dumps({'status': 'begin', 'pollutant': pol.upper(), 'progress': 0, 'message': f'Starting {pol.upper()}'})}\n\n"
        try:
            df = df_full[["waktu", pol]].rename(columns={"waktu": "ds", pol: "y"}).dropna(subset=["y"])
            total_combos = len(param_grid)
            best_mape, best_model = float("inf"), None

            for i, (cp, ss, hs, w, y) in enumerate(param_grid, start=1):
                progress = round((i / total_combos) * 60, 2)
                yield f"data: {json.dumps({'status': 'progress', 'pollutant': pol.upper(), 'progress': progress, 'message': f'Training'})}\n\n"
                try:
                    model = Prophet(
                        yearly_seasonality=y,
                        weekly_seasonality=w,
                        holidays=holidays,
                        changepoint_prior_scale=cp,
                        seasonality_prior_scale=ss,
                        holidays_prior_scale=hs
                    )
                    model.add_seasonality("monthly", period=30.5, fourier_order=5)
                    model.fit(df)
                    cv = cross_validation(model, initial="180 days", period="180 days", horizon="60 days")
                    mape = mean_absolute_percentage_error(cv["y"], cv["yhat"])
                    if mape < best_mape:
                        best_mape, best_model = mape, model
                except Exception:
                    continue

            if not best_model:
                yield f"data: {json.dumps({'status': 'error', 'pollutant': pol.upper(), 'message': 'No valid model'})}\n\n"
                continue

            yield f"data: {json.dumps({'status': 'progress', 'pollutant': pol.upper(), 'progress': 75, 'message': f'Generating forecast for {pol.upper()}'})}\n\n"

            forecast = best_model.predict(best_model.make_future_dataframe(periods=30))
            result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30)
            result["ds"] = result["ds"].dt.strftime("%Y-%m-%d")

            # Simpan ke DB
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM forecast_{pol}_with_parameters_data")
            insert_sql = "INSERT INTO forecast_{0}_with_parameters_data (ds, yhat, yhat_lower, yhat_upper) VALUES (%s, %s, %s, %s)".format(pol)
            for _, row in result.iterrows():
                cursor.execute(insert_sql, (
                    datetime.strptime(row["ds"], "%Y-%m-%d").date(),
                    float(row["yhat"]), float(row["yhat_lower"]), float(row["yhat_upper"])
                ))
            conn.commit(); cursor.close(); conn.close()

            yield f"data: {json.dumps({'status': 'done', 'pollutant': pol.upper(), 'progress': 100, 'message': f'{pol.upper()} completed (MAPE={best_mape:.4f})'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'pollutant': pol.upper(), 'message': str(e)})}\n\n"
