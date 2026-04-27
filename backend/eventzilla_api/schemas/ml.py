from pydantic import BaseModel, Field


class PricePredictRequest(BaseModel):
    price: float
    budget: float
    marketing_spend: float
    new_beneficiaries: float
    reservations: float
    nb_events: float
    avg_spent_user: float
    type: str | None = None
    status: str | None = None


class InputForecast(BaseModel):
    category_name: str
    horizon: int = Field(default=6, ge=1, le=24)


class InputFidelisation(BaseModel):
    price: float
    budget: float
    final_price: float
    rating: float
    visitors: float
    marketing_spend: float
    price_budget_ratio: float
    margin: float
    has_complaint: int
    type_encoded: int
    season_encoded: int
    is_weekend: int
    month: int


class InputSentiment(BaseModel):
    text: str


class ForecastPoint(BaseModel):
    date: str
    value: float


class ForecastMetrics(BaseModel):
    mape: float
    mae: float
    rmse: float
    train_points: int
    test_points: int


class ForecastResponse(BaseModel):
    status: str
    category: str
    model: str
    history: list[ForecastPoint]
    forecast: list[ForecastPoint]
    metrics: ForecastMetrics


class SentimentResponse(BaseModel):
    status: str
    text: str
    polarity: float
    label: str


class RecommendationResponse(BaseModel):
    status: str
    beneficiary_id: int | None = None
    recommendations: list[int]
    type: str
    scores: list[float] = []


class AnomalyRecord(BaseModel):
    price: float
    budget: float
    final_price: float
    rating: float
    visitors: float
    marketing_spend: float
    ano_score: float


class AnomalyResponse(BaseModel):
    status: str
    total_count: int
    anomaly_count: int
    sample_anomalies: list[AnomalyRecord]
    algorithm: str
    contamination: float


class DeepLearningResponse(BaseModel):
    status: str
    model: str
    prediction: int
    confidence: float
    note: str
    accuracy: float
    f1_score: float
    auc: float
    iterations: int
