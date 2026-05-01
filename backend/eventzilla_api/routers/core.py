from functools import lru_cache

from fastapi import APIRouter
import pandas as pd

from ..schemas.ml import InputFidelisation, InputForecast, InputSentiment, PricePredictRequest
from ..services import MLService
from ..db import get_engine


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


@router.post("/chatbot")
def chatbot(body: dict) -> dict:
    message = (body.get("message") or "").lower().strip()
    engine = get_engine()

    rules = [
        # Reservations count
        (["how many reservation", "count reserv", "total reserv", "number of reserv"],
         "SELECT COUNT(*) AS total_reservations FROM dim_reservation",
         "kpi"),

        # Lowest rated providers — must come BEFORE top providers rule
        (["lowest rating", "worst provider", "low rating", "poor rating"],
         """SELECT p.name_provider, ROUND(AVG(f.rating)::numeric,2) AS avg_rating,
                   COUNT(*) AS events
            FROM fact_suivi_event f
            JOIN dim_provider p ON f.sk_provider = p.sk_provider
            GROUP BY p.name_provider HAVING COUNT(*) > 2
            ORDER BY avg_rating ASC LIMIT 5""",
         "chart"),

        # Top providers
        (["top", "provider", "best provider", "most reserv"],
         """SELECT p.name_provider, SUM(f.reservations) AS total_reservations,
                   ROUND(AVG(f.rating)::numeric,2) AS avg_rating
            FROM fact_suivi_event f
            JOIN dim_provider p ON f.sk_provider = p.sk_provider
            GROUP BY p.name_provider ORDER BY total_reservations DESC LIMIT 5""",
         "chart"),

        # Reservation status
        (["status", "reservation status", "by status"],
         """SELECT status, COUNT(*) AS count
            FROM dim_reservation GROUP BY status ORDER BY count DESC""",
         "chart"),

        # Average rating
        (["average rating", "avg rating", "mean rating", "rating"],
         """SELECT ROUND(AVG(rating)::numeric,2) AS avg_rating,
                   MIN(rating)::numeric AS min_rating,
                   MAX(rating)::numeric AS max_rating
            FROM fact_suivi_event WHERE rating IS NOT NULL""",
         "kpi"),

        # Category with most services
        (["category", "most service", "which category", "event category"],
         """SELECT c.category_name, COUNT(*) AS event_count,
                   ROUND(AVG(f.rating)::numeric,2) AS avg_rating
            FROM fact_suivi_event f
            JOIN "DIM_category" c ON f.category_id = c.category_id
            GROUP BY c.category_name ORDER BY event_count DESC LIMIT 5""",
         "chart"),

        # Visitor trend by month
        (["visitor", "trend", "by month", "monthly"],
         """SELECT d.year, d.month, SUM(f.visitors) AS total_visitors
            FROM fact_suivi_event f
            JOIN dim_date d ON f.date_event_fk = d.date_id
            GROUP BY d.year, d.month ORDER BY d.year, d.month LIMIT 12""",
         "chart"),

        # Total revenue
        (["revenue", "total revenue", "income", "earnings"],
         """SELECT ROUND(SUM(final_price)::numeric,2) AS total_revenue,
                   ROUND(AVG(final_price)::numeric,2) AS avg_revenue_per_event,
                   COUNT(*) AS total_events
            FROM fact_suivi_event WHERE final_price IS NOT NULL""",
         "kpi"),

        # Event types
        (["event type", "type of event", "wedding", "corporate", "party"],
         """SELECT event_type, COUNT(*) AS count,
                   ROUND(AVG(rating)::numeric,2) AS avg_rating
            FROM view_event_analysis
            GROUP BY event_type ORDER BY count DESC""",
         "chart"),

        # Complaints
        (["complaint", "issue", "problem"],
         """SELECT complaint_subject, complaint_status, COUNT(*) AS count
            FROM view_event_analysis
            WHERE complaint_subject IS NOT NULL
            GROUP BY complaint_subject, complaint_status ORDER BY count DESC LIMIT 10""",
         "chart"),

        # Budget vs price
        (["budget", "price", "cost"],
         """SELECT ROUND(AVG(budget)::numeric,2) AS avg_budget,
                   ROUND(AVG(price)::numeric,2) AS avg_price,
                   ROUND(AVG(final_price)::numeric,2) AS avg_final_price
            FROM fact_suivi_event""",
         "kpi"),

        # Top events
        (["top event", "best event", "popular event"],
         """SELECT event_title, event_type, rating, visitors, final_price
            FROM view_event_analysis
            ORDER BY rating DESC, visitors DESC LIMIT 5""",
         "chart"),

        # Season analysis
        (["season", "saison", "summer", "winter", "spring"],
         """SELECT saison, COUNT(*) AS events,
                   ROUND(AVG(rating)::numeric,2) AS avg_rating,
                   SUM(visitors) AS total_visitors
            FROM view_event_analysis
            GROUP BY saison ORDER BY events DESC""",
         "chart"),
    ]

    try:
        for keywords, sql, rtype in rules:
            if any(k in message for k in keywords):
                df = pd.read_sql(sql, engine)
                records = df.to_dict(orient="records")
                if rtype == "kpi" and records:
                    parts = [f"{k.replace('_',' ').title()}: **{v}**" for k, v in records[0].items()]
                    reply = " | ".join(parts)
                else:
                    reply = f"Here are the results ({len(records)} rows):"
                return {"reply": reply, "sql": sql, "data": records, "type": rtype, "status": "ok"}

        return {
            "reply": "I can answer questions about: reservations, providers, ratings, revenue, visitors, categories, event types, complaints, budget, seasons. Try: 'Top 5 providers', 'Total revenue', 'Show visitor trend by month'.",
            "sql": None, "data": [], "type": "general", "status": "ok"
        }
    except Exception as e:
        return {
            "reply": f"Database error: {str(e)[:200]}",
            "sql": None, "data": [], "type": "general", "status": "error"
        }


@router.post("/train/price")
def train_price() -> dict:
    return get_service().train_price()


@router.post("/train/fidelisation")
def train_fidelisation() -> dict:
    return get_service().train_fidelisation()


@router.post("/train/forecast")
def train_forecast() -> dict:
    return get_service().train_forecast()


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

