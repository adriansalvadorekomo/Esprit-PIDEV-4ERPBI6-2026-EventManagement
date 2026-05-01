import os
import sys
import glob
import pickle
import re
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# Make eventzilla_api importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="EventZella API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── /predict model loading ───────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _find_latest(pattern):
    files = sorted(glob.glob(os.path.join(BASE, pattern)))
    return files[-1] if files else None

model = scaler = columns = None
try:
    model_path  = _find_latest("models/rf_model_*.pkl") or os.path.join(BASE, "rf_model.pkl")
    scaler_path = _find_latest("models/scaler_*.pkl")   or os.path.join(BASE, "scaler.pkl")
    columns_path = os.path.join(BASE, "columns.pkl")
    if os.path.exists(model_path):
        model  = _load_pickle(model_path);  print(f"✅ Model loaded: {model_path}")
    if os.path.exists(scaler_path):
        scaler = _load_pickle(scaler_path); print(f"✅ Scaler loaded: {scaler_path}")
    if os.path.exists(columns_path):
        columns = _load_pickle(columns_path)
except Exception as e:
    print(f"❌ Model load error: {e}")

# ── Cluster model loading ────────────────────────────────────
kmeans_cl = dbscan_cl = scaler_cl = pca_cl = freq_map_cl = cluster_names_cl = None
try:
    kmeans_cl       = joblib.load(os.path.join(BASE, "kmeans_model.joblib"))
    dbscan_cl       = joblib.load(os.path.join(BASE, "dbscan_model.joblib"))
    scaler_cl       = joblib.load(os.path.join(BASE, "scaler.joblib"))
    pca_cl          = joblib.load(os.path.join(BASE, "pca.joblib"))
    freq_map_cl     = joblib.load(os.path.join(BASE, "freq_map.joblib"))
    cluster_names_cl = joblib.load(os.path.join(BASE, "cluster_names.joblib"))
    print("✅ Cluster models loaded")
except Exception as e:
    print(f"❌ Cluster model load error: {e}")

# ── Chatbot: DB + Groq setup ─────────────────────────────────
from settings import DATABASE_URL as DB_URL

DB_SCHEMA = """
You are a senior BI Data Analyst for EventZella, an event management platform.
Your ONLY job is to return a single raw SQL query. No explanation. No markdown. No prose. No backticks.

Database tables and their columns:
- stg_beneficiary(id_beneficiary, first_name, last_name, email, phone)
- stg_category(id_category, name)
- stg_subcategory(id_subcategory, name, id_category)
- stg_provider(id_provider, name, service_type, email, phone, city)
- stg_service(id_service, title, price, description, id_provider, id_subcategory)
- stg_event(id_event, title, event_date, budget, type, id_beneficiary)
- stg_reservation(id_reservation, id_service, id_event, reservation_date, status, final_price)
- stg_evaluation(id_evaluation, id_reservation, rating, comment)
- stg_complaint(id_complaint, subject, description, status, id_beneficiary, id_provider)
- stg_marketing_spend(id, month, marketing_spend, new_beneficiaries)
- stg_visitors(id, date, visitors, reservations)

CRITICAL RULES:
1. Always use table aliases.
2. All numeric columns are stored as VARCHAR. Cast with ::NUMERIC before any math.
3. Return ONLY the raw SQL string. Nothing else.
4. For aggregation questions, use COUNT(*), SUM(col::NUMERIC), AVG(col::NUMERIC).
5. Always alias result columns clearly.
6. For top/bottom items, use ORDER BY and LIMIT.
7. For chart-worthy data, return multiple rows with a label column and a value column.
"""

INSIGHT_PROMPT = """
You are a concise BI analyst. Given a SQL query result, provide ONE short business insight (2-3 sentences max).
Be direct. No filler. Focus on what the number means for the business.
"""

def _get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    from groq import Groq
    return Groq(api_key=api_key)

def extract_sql(text_response: str) -> str:
    clean = re.sub(r"```sql|```", "", text_response).strip()
    lines = clean.splitlines()
    sql_lines, started = [], False
    for line in lines:
        upper = line.strip().upper()
        if not started and any(upper.startswith(kw) for kw in ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE"]):
            started = True
        if started:
            sql_lines.append(line)
    return "\n".join(sql_lines).strip() if sql_lines else clean

def classify_question(q: str) -> str:
    q_lower = q.lower()
    chart_kw = ["trend", "over time", "by month", "by category", "by city", "top", "breakdown", "distribution", "compare", "per"]
    kpi_kw   = ["total", "sum", "count", "average", "avg", "how many", "how much", "visitors", "reservations", "budget", "revenue", "rating", "complaints", "providers", "beneficiaries"]
    if any(k in q_lower for k in chart_kw): return "chart"
    if any(k in q_lower for k in kpi_kw):   return "kpi"
    return "general"

def get_sql_from_ai(client, question: str, q_type: str):
    hint = " Return multiple rows with a label column and a numeric value column." if q_type == "chart" else ""
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": DB_SCHEMA},
                {"role": "user",   "content": question + hint}
            ],
            temperature=0, max_tokens=500
        )
        return extract_sql(resp.choices[0].message.content.strip())
    except Exception as e:
        print(f"SQL gen error: {e}")
        return None

def run_sql(sql: str):
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn)
    except Exception as e:
        print(f"SQL exec error: {e}")
        return None

def get_insight(client, question: str, result_summary: str):
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": INSIGHT_PROMPT},
                {"role": "user",   "content": f"Question: {question}\nResult: {result_summary}"}
            ],
            temperature=0.3, max_tokens=150
        )
        return resp.choices[0].message.content.strip()
    except:
        return None

def format_number(val) -> str:
    if isinstance(val, float):
        if val >= 1_000_000: return f"{val/1_000_000:.2f}M"
        if val >= 1_000:     return f"{val:,.0f}"
        return f"{val:,.2f}"
    if isinstance(val, int):
        if val >= 1_000_000: return f"{val/1_000_000:.2f}M"
        return f"{val:,}"
    return str(val)

# ── Schemas ──────────────────────────────────────────────────
class PredictRequest(BaseModel):
    attendees: int
    duration: int

class ChatRequest(BaseModel):
    message: str

class ClusterRequest(BaseModel):
    budget: float = 0
    price: float = 0
    final_price: float = 0
    rating: float = 0
    visitors: float = 0
    complaint_status: str | None = None
    complaint_subject: str | None = None
    event_type: str | None = None
    reservation_status: str | None = None
    algo: str = "dbscan"

# ── Routes ───────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "EventZella API is running"}


@app.post("/predict")
def predict(body: PredictRequest):
    print(f"📥 Input: attendees={body.attendees}, duration={body.duration}")
    if model is None:
        return {"prediction": 0, "status": "model_not_loaded"}
    try:
        if columns is not None:
            row = dict.fromkeys(columns, 0)
            if "reservations" in row: row["reservations"] = body.attendees
            if "price" in row:        row["price"] = body.duration * 10
            X = pd.DataFrame([row])[columns].values
        else:
            X = np.array([[body.attendees, body.duration]])
        if scaler is not None:
            X = scaler.transform(X)
        result = model.predict(X)
        prediction = int(result[0])
        print(f"📤 Prediction: {prediction}")
        return {"prediction": prediction, "status": "success"}
    except Exception as e:
        print(f"❌ Predict error: {e}")
        return {"prediction": 0, "status": "error"}


@app.post("/predict-cluster")
def predict_cluster(body: ClusterRequest):
    if scaler_cl is None or pca_cl is None or kmeans_cl is None or dbscan_cl is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Cluster models not loaded")

    algo = body.algo.lower()
    cols = [
        'budget', 'price', 'final_price', 'rating', 'visitors',
        'complaint_status_bin', 'complaint_subject_freq',
        'event_type_Corporate Event', 'event_type_Private Party',
        'event_type_Wedding', 'reservation_status_cancelled',
        'reservation_status_confirmed', 'reservation_status_pending'
    ]
    row = {
        'budget': body.budget, 'price': body.price, 'final_price': body.final_price,
        'rating': body.rating, 'visitors': body.visitors,
        'complaint_status_bin': 1 if body.complaint_status == 'open' else 0,
        'complaint_subject_freq': (freq_map_cl or {}).get(body.complaint_subject or '', 0.0),
        'event_type_Corporate Event': 1 if body.event_type == 'Corporate Event' else 0,
        'event_type_Private Party': 1 if body.event_type == 'Private Party' else 0,
        'event_type_Wedding': 1 if body.event_type == 'Wedding' else 0,
        'reservation_status_cancelled': 1 if body.reservation_status == 'cancelled' else 0,
        'reservation_status_confirmed': 1 if body.reservation_status == 'confirmed' else 0,
        'reservation_status_pending': 1 if body.reservation_status == 'pending' else 0,
    }
    df_input = pd.DataFrame([row])[cols]
    print(f"Colonnes envoyées au scaler : {df_input.columns.tolist()}")
    X_scaled = scaler_cl.transform(df_input)
    X_pca = pca_cl.transform(X_scaled)

    if algo == 'dbscan':
        samples = dbscan_cl.components_
        labels = dbscan_cl.labels_[dbscan_cl.core_sample_indices_]
        nn = NearestNeighbors(n_neighbors=1).fit(samples)
        _, indices = nn.kneighbors(X_pca)
        cluster_id = int(labels[indices[0][0]])
    else:
        cluster_id = int(kmeans_cl.predict(X_pca)[0])

    names = cluster_names_cl or {0: "Client Premium", 1: "Client Potentiel", 2: "Client à Risque", -1: "Inclassable"}
    return {
        "cluster_id": cluster_id,
        "cluster_name": names.get(cluster_id, "Inconnu"),
        "algorithm": algo,
        "status": "success",
    }


# ── Delegate to eventzilla_api ───────────────────────────────
try:
    from eventzilla_api.app import create_app as _create_full_app
    from eventzilla_api.services import MLService
    from eventzilla_api.schemas.ml import (
        InputFidelisation, PricePredictRequest as _PricePredictRequest,
        InputForecast, InputSentiment
    )
    from eventzilla_api.db import get_engine as _get_engine
    from fastapi import Query

    _svc = None
    def _get_svc():
        global _svc
        if _svc is None:
            _svc = MLService()
        return _svc

    @app.get("/categories")
    def get_categories():
        return _get_svc().get_categories()

    @app.post("/predict/fidelisation")
    def predict_fidelisation(data: InputFidelisation):
        return _get_svc().predict_fidelisation(data)

    @app.post("/predict/price")
    def predict_price_full(data: _PricePredictRequest):
        return _get_svc().predict_price(data)

    @app.post("/train/price")
    def train_price():
        return _get_svc().train_price()

    @app.post("/train/fidelisation")
    def train_fidelisation():
        return _get_svc().train_fidelisation()

    @app.post("/predict/forecast")
    def predict_forecast(data: InputForecast):
        return _get_svc().predict_forecast(data)

    @app.post("/predict/sentiment")
    def predict_sentiment(data: InputSentiment):
        return _get_svc().predict_sentiment(data)

    @app.post("/recommend/events")
    def recommend_events(beneficiary_id: int = Query(..., ge=1), n_reco: int = Query(default=5, ge=1, le=10)):
        return _get_svc().recommend_events(beneficiary_id=beneficiary_id, n_reco=n_reco)

    @app.post("/predict/anomalies")
    def detect_anomalies():
        return _get_svc().detect_anomalies()

    @app.post("/predict/deep-learning")
    def predict_deep_learning(data: InputFidelisation):
        return _get_svc().predict_deep_learning(data)

    print("eventzilla_api routes registered OK")
except Exception as _e:
    print(f"eventzilla_api import failed: {_e}")


@app.post("/chatbot")
def chatbot(body: ChatRequest):
    client = _get_groq_client()
    if client is None:
        return {"reply": "Chatbot API key is missing.", "sql": None, "data": [], "type": "general", "status": "error"}

    question = body.message.strip()
    q_type   = classify_question(question)
    sql      = get_sql_from_ai(client, question, q_type)

    if not sql:
        return {"reply": "Sorry, I couldn't process that request.", "sql": None, "data": [], "type": q_type, "status": "error"}

    df = run_sql(sql)

    if df is None or df.empty:
        return {"reply": "The query ran but returned no results.", "sql": sql, "data": [], "type": q_type, "status": "success"}

    data = df.head(50).to_dict(orient="records")

    # Build reply
    if q_type == "kpi" or (df.shape[0] == 1 and df.shape[1] == 1):
        val        = df.iloc[0, 0]
        col_name   = df.columns[0].replace("_", " ").title()
        formatted  = format_number(val) if isinstance(val, (int, float)) else str(val)
        insight    = get_insight(client, question, f"{col_name}: {formatted}")
        reply      = f"{col_name}: {formatted}"
        if insight:
            reply += f"\n\n{insight}"
    else:
        summary = df.head(5).to_string(index=False)
        insight = get_insight(client, question, summary)
        reply   = f"Found {len(df)} results."
        if insight:
            reply += f"\n\n{insight}"

    return {"reply": reply, "sql": sql, "data": data, "type": q_type, "status": "success"}
