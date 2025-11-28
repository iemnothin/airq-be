from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router

app = FastAPI(
    title="AirQ API",
    description="""
    🌤️ **AirQ API — Air Quality Forecasting Platform**

    This API handles air quality data ingestion, cleaning, 
    outlier handling, and forecasting (using Facebook Prophet).

    **Main Features:**
    - Upload & store air quality data
    - Detect and interpolate outliers
    - Train Prophet models (basic & advanced)
    - Retrieve or clear forecast results
    """,
    version="1.0.0",
)

# Enable CORS (optional but recommended for frontend)
origins = [
    "https://airq.abiila.com",
    "http://airq.abiila.com",
    "https://api-airq.abiila.com",
    "http://api-airq.abiila.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register router
app.include_router(router)


@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to AirQ API 🚀"}


# Run command:
# uvicorn main:app --reload
