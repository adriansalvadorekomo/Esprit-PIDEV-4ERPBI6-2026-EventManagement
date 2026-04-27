from functools import lru_cache

from fastapi import APIRouter, Query

from ..schemas.ml import InputFidelisation
from ..services import MLService


router = APIRouter()


@lru_cache(maxsize=1)
def get_service() -> MLService:
    return MLService()


@router.post("/recommend/events")
def recommend_events(
    beneficiary_id: int = Query(..., ge=1),
    n_reco: int = Query(default=5, ge=1, le=10),
):
    return get_service().recommend_events(beneficiary_id=beneficiary_id, n_reco=n_reco)


@router.post("/predict/anomalies")
def detect_anomalies():
    return get_service().detect_anomalies()


@router.post("/predict/deep-learning")
def predict_deep_learning(data: InputFidelisation):
    return get_service().predict_deep_learning(data)
