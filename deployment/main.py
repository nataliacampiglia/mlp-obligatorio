from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from services.prediction import load_model, predict, get_country_options, is_ready, get_model_version


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Predictor de Inflación en Alimentos", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class PredictRequest(BaseModel):
    country: str
    year: int
    month: int  # 1-12


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/status")
def status():
    return {"ready": is_ready(), "version": get_model_version()}


@app.get("/countries")
def get_countries():
    return {"countries": get_country_options()}


@app.post("/predict")
def run_prediction(req: PredictRequest):
    value = predict(req.country, req.year, req.month)
    return {
        "country": req.country,
        "year": req.year,
        "month": req.month,
        "inflation": value,
    }
