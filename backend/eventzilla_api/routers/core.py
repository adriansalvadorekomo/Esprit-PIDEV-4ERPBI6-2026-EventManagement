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




def _ask_groq(message: str, db_context: str) -> str:
    import requests as _req
    import settings as _s
    GROQ_API_KEY = _s.GROQ_API_KEY
    system_prompt = (
        "You are EventZella AI, a Business Intelligence Copilot for an event management company.\n"
        "Analyze the data and provide clear insights, anomaly detection, and actionable recommendations.\n"
        "Be concise and professional. Answer in under 4 sentences when possible.\n\n"
        "=== Database Context ===\n" + db_context
    )
    try:
        resp = _req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": message},
                ],
                "temperature": 0.3,
            },
            timeout=30,
        )
        data = resp.json()
        if "choices" not in data:
            return "AI error: " + data.get("error", {}).get("message", str(data))
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        return f"AI error: {exc}"


@router.post("/chatbot")
def chatbot(body: dict) -> dict:
    message = (body.get("message") or "").strip()
    engine  = get_engine()

    # Fetch live KPI context
    db_context = ""
    try:
        kpi_df = pd.read_sql(
            """SELECT ROUND(SUM(final_price)::numeric,2) AS total_revenue,
                      ROUND(AVG(final_price)::numeric,2) AS avg_revenue,
                      ROUND(AVG(rating)::numeric,2)      AS avg_rating,
                      COUNT(*)                           AS total_events,
                      SUM(visitors)                      AS total_visitors,
                      SUM(reservations)                  AS total_reservations
               FROM fact_suivi_event WHERE final_price IS NOT NULL""",
            engine,
        )
        if not kpi_df.empty:
            db_context = "\n".join(f"{k}: {v}" for k, v in kpi_df.iloc[0].items())
    except Exception:
        db_context = "Database KPIs unavailable."

    # Fetch relevant data rows for chart questions
    data_records: list = []
    sql_used = None
    rtype = "general"

    # (keywords, sql, type, chart_type)
    rules = [
        (["how many reservation", "count reserv", "total reserv", "number of reserv"],
         "SELECT COUNT(*) AS total_reservations FROM dim_reservation", "kpi", None),
        (["lowest rating", "worst provider", "low rating", "poor rating"],
         """SELECT p.name_provider, ROUND(AVG(f.rating)::numeric,2) AS avg_rating, COUNT(*) AS events
            FROM fact_suivi_event f JOIN dim_provider p ON f.sk_provider = p.sk_provider
            GROUP BY p.name_provider HAVING COUNT(*) > 2 ORDER BY avg_rating ASC LIMIT 5""", "chart", "bar"),
        (["top", "provider", "best provider", "most reserv"],
         """SELECT p.name_provider, SUM(f.reservations) AS total_reservations,
                   ROUND(AVG(f.rating)::numeric,2) AS avg_rating
            FROM fact_suivi_event f JOIN dim_provider p ON f.sk_provider = p.sk_provider
            GROUP BY p.name_provider ORDER BY total_reservations DESC LIMIT 5""", "chart", "horizontalBar"),
        (["status", "reservation status", "by status"],
         "SELECT status, COUNT(*) AS count FROM dim_reservation GROUP BY status ORDER BY count DESC", "chart", "pie"),
        (["average rating", "avg rating", "mean rating", "rating"],
         """SELECT ROUND(AVG(rating)::numeric,2) AS avg_rating,
                   MIN(rating)::numeric AS min_rating, MAX(rating)::numeric AS max_rating
            FROM fact_suivi_event WHERE rating IS NOT NULL""", "kpi", None),
        (["category", "most service", "which category", "event category"],
         """SELECT c.category_name, COUNT(*) AS event_count, ROUND(AVG(f.rating)::numeric,2) AS avg_rating
            FROM fact_suivi_event f JOIN "DIM_category" c ON f.category_id = c.category_id
            GROUP BY c.category_name ORDER BY event_count DESC LIMIT 5""", "chart", "bar"),
        (["visitor", "trend", "by month", "monthly"],
         """SELECT d.year, d.month, SUM(f.visitors) AS total_visitors
            FROM fact_suivi_event f JOIN dim_date d ON f.date_event_fk = d.date_id
            GROUP BY d.year, d.month ORDER BY d.year, d.month LIMIT 12""", "chart", "line"),
        (["revenue", "total revenue", "income", "earnings"],
         """SELECT ROUND(SUM(final_price)::numeric,2) AS total_revenue,
                   ROUND(AVG(final_price)::numeric,2) AS avg_revenue_per_event, COUNT(*) AS total_events
            FROM fact_suivi_event WHERE final_price IS NOT NULL""", "kpi", None),
        (["event type", "type of event", "wedding", "corporate", "party"],
         """SELECT event_type, COUNT(*) AS count, ROUND(AVG(rating)::numeric,2) AS avg_rating
            FROM view_event_analysis GROUP BY event_type ORDER BY count DESC""", "chart", "pie"),
        (["complaint", "issue", "problem"],
         """SELECT complaint_subject, complaint_status, COUNT(*) AS count
            FROM view_event_analysis WHERE complaint_subject IS NOT NULL
            GROUP BY complaint_subject, complaint_status ORDER BY count DESC LIMIT 10""", "chart", "bar"),
        (["budget", "price", "cost"],
         """SELECT ROUND(AVG(budget)::numeric,2) AS avg_budget, ROUND(AVG(price)::numeric,2) AS avg_price,
                   ROUND(AVG(final_price)::numeric,2) AS avg_final_price FROM fact_suivi_event""", "kpi", None),
        (["season", "saison", "summer", "winter", "spring"],
         """SELECT saison, COUNT(*) AS events, ROUND(AVG(rating)::numeric,2) AS avg_rating,
                   SUM(visitors) AS total_visitors
            FROM view_event_analysis GROUP BY saison ORDER BY events DESC""", "chart", "bar"),
    ]

    msg_lower = message.lower()
    chart_type = None
    try:
        for keywords, sql, rt, ct in rules:
            if any(k in msg_lower for k in keywords):
                df = pd.read_sql(sql, engine)
                data_records = df.to_dict(orient="records")
                sql_used = sql
                rtype = rt
                chart_type = ct
                db_context += "\n\nRelevant query results:\n" + df.head(10).to_string(index=False)
                break
    except Exception:
        pass

    reply = _ask_groq(message, db_context)

    return {
        "reply":      reply,
        "sql":        sql_used,
        "data":       data_records,
        "type":       rtype,
        "chart_type": chart_type,
        "status":     "ok",
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

