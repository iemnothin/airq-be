from pydantic import BaseModel
from typing import Optional

class AirQuality(BaseModel):
    id: Optional[int]
    waktu: str
    pm10: float
    pm25: float
    so2: float
    co: float
    o3: float
    no2: float
    hc: float
    kelembaban: Optional[float]
    suhu: Optional[float]

class ForecastResult(BaseModel):
    ds: str
    yhat: float
    yhat_lower: float
    yhat_upper: float
