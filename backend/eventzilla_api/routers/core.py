from functools import lru_cache

from fastapi import APIRouter

from ..schemas.ml import InputFidelisation, InputForecast, InputSentiment, PricePredictRequest
from ..services import MLService


router = APIRouter()


@lru_cache(maxsize=1)
def get_service() -> MLService:
    return MLService()


@router.get("/")
def home() -> dict:
    return {
        "message": "EventZilla API running",
        "docs": "/docs",
        "health": "/health",
    }


@router.post("/train/price")
def train_price() -> dict:
    return get_service().train_price()


@router.post("/train/fidelisation")
def train_fidelisation() -> dict:
    return get_service().train_fidelisation()


@router.post("/predict/price")
def predict_price(data: PricePredictRequest) -> dict:
    return get_service().predict_price(data)


@router.post("/predict/fidelisation")
def predict_fidelisation(data: InputFidelisation) -> dict:
    return get_service().predict_fidelisation(data)


@router.get("/categories")
def get_categories() -> list[str]:
    return get_service().get_categories()


@router.post("/predict/forecast")
def predict_forecast(data: InputForecast):
    return get_service().predict_forecast(data)


@router.post("/predict/sentiment")
def predict_sentiment(data: InputSentiment):
    return get_service().predict_sentiment(data)

