from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os
import subprocess
import joblib
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE

app = FastAPI()

# =============================
# CORS — Angular
# =============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for development
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# MLFLOW CONFIG
# =============================
@app.on_event("startup")
def init_mlflow():
    # mlflow.set_tracking_uri("http://127.0.0.1:5000") # Port 5000 is used by the Flask app
    mlflow.set_experiment("ml_pipeline")

# =============================
# DB ENGINE — pour fidélisation
# =============================
engine = create_engine(
    "postgresql+psycopg2://postgres:1400@localhost:5432/DW_event?client_encoding=utf8",
    pool_size=2,
    max_overflow=0,
    pool_pre_ping=True,
    pool_recycle=300
)

# =============================
# TON MODÈLE — Régression final_price
# =============================
rf_regression_model  = None
rf_regression_scaler = None
last_loaded          = 0

def load_regression_model():
    global rf_regression_model, rf_regression_scaler, last_loaded

    model_path  = "rf_model.pkl"   # ✅ nom unique
    scaler_path = "scaler.pkl"  # ✅ nom unique

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise HTTPException(
            status_code=404,
            detail="Modèle régression introuvable. Lance /train/price d'abord."
        )

    current_time = os.path.getmtime(model_path)
    if rf_regression_model is None or current_time != last_loaded:
        with open(model_path, "rb") as f:
            rf_regression_model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            rf_regression_scaler = pickle.load(f)
        last_loaded = current_time
        print("✅ Modèle régression reloaded")

# =============================
# MODÈLE AMI — Classification Fidélisation
# =============================
try:
    lr_model        = joblib.load("lr_model.pkl")
    rf_classif      = joblib.load("rf_classif_model.pkl")  # ✅ nom unique
    xgb_model       = joblib.load("xgb_model.pkl")
    scaler_classif  = joblib.load("scaler_classif.pkl")
    le_type         = joblib.load("le_type.pkl")
    le_season       = joblib.load("le_season.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    print("✅ Modèles fidélisation chargés")
except Exception as e:
    print(f"⚠️ Modèles fidélisation non chargés : {e}")
    lr_model = rf_classif = xgb_model = None
    scaler_classif = le_type = le_season = None
    feature_columns = [
        'price', 'budget', 'final_price', 'rating', 'visitors',
        'marketing_spend', 'price_budget_ratio', 'margin',
        'has_complaint', 'type_encoded', 'season_encoded',
        'is_weekend', 'month'
    ]

# =============================
# INPUT — Fidélisation
# =============================

class InputForecast(BaseModel):
    category_name: str
    horizon: int = 6

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

# =============================
# HOME
# =============================
@app.get("/")
def home():
    return {
        "message": "✅ API ML running",
        "endpoints": {
            "POST /train/price":           "Entraîner modèle régression",
            "POST /train/fidelisation":    "Entraîner modèle fidélisation",
            "POST /predict/price":         "Prédire final_price",
            "POST /predict/fidelisation":  "Prédire fidélisation",
            "POST /predict/fidelisation/batch": "Prédire fidélisation en batch"
        }
    }

# =============================
# TRAIN — Régression (ton train.py)
# =============================
@app.post("/train/price")
def train_price():
    try:
        result = subprocess.run(
            ["python", "train_reg.py"],
            capture_output=True, text=True, check=True
        )
        load_regression_model()
        return {
            "status": "success",
            "model":  "regression_final_price",
            "output": result.stdout
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail={
            "message": "Erreur training régression",
            "details": e.stderr
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================
# TRAIN — Fidélisation (inline)
# =============================
@app.post("/train/fidelisation")
def train_fidelisation():
    try:
        global lr_model, rf_classif, xgb_model
        global scaler_classif, le_type, le_season, feature_columns

        query = """
            SELECT f.sk_beneficiary, f.event_sk, f.price, f.budget, f.final_price,
                   f.rating, f.visitors, f.marketing_spend, f.new_beneficiaries,
                   f.id_complaint, f.reservations,
                   convert_from(convert_to(e.type, 'LATIN1'), 'UTF8') as type,
                   e.event_date,
                   convert_from(convert_to(r.status, 'LATIN1'), 'UTF8') as status,
                   d.quarter, d.year
            FROM fact_suivi_event f
            LEFT JOIN dim_event e ON f.event_sk = e.event_sk
            LEFT JOIN dim_reservation r ON f.reservation_sk = r.reservation_sk
            LEFT JOIN dim_date d ON f.date_event_fk = d.date_id
        """
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        df['event_date'] = pd.to_datetime(df['event_date'])
        df = df.sort_values(['sk_beneficiary', 'event_date'])
        df['status'] = df['status'].replace({'cancellé': 'cancelled'})
        df['month']       = df['event_date'].dt.month
        df['day_of_week'] = df['event_date'].dt.dayofweek
        df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)

        def get_season(month):
            if month in [12, 1, 2]:  return 'hiver'
            elif month in [3, 4, 5]: return 'printemps'
            elif month in [6, 7, 8]: return 'ete'
            else:                    return 'automne'

        df['season']             = df['month'].apply(get_season)
        df['price_budget_ratio'] = df['price'] / (df['budget'] + 0.01)
        df['margin']             = df['final_price'] - df['price']
        df['has_complaint']      = (~df['id_complaint'].isna()).astype(int)

        le_type   = LabelEncoder()
        le_season = LabelEncoder()
        df['type_encoded']   = le_type.fit_transform(df['type'].fillna('inconnu'))
        df['season_encoded'] = le_season.fit_transform(df['season'])

        def create_loyalty_target(df, months_window=6):
            loyalty_map = {}
            for ben in df['sk_beneficiary'].unique():
                ben_data   = df[df['sk_beneficiary'] == ben].sort_values('event_date')
                if len(ben_data) <= 1:
                    loyalty_map[ben] = 0
                    continue
                first_date = ben_data['event_date'].iloc[0]
                later      = ben_data[
                    (ben_data['event_date'] > first_date) &
                    (ben_data['event_date'] <= first_date + pd.DateOffset(months=months_window))
                ]
                loyalty_map[ben] = 1 if len(later) >= 1 else 0
            df['is_loyal'] = df['sk_beneficiary'].map(loyalty_map)
            return df

        df = create_loyalty_target(df, months_window=6)

        feature_columns = [
            'price', 'budget', 'final_price', 'rating', 'visitors',
            'marketing_spend', 'price_budget_ratio', 'margin',
            'has_complaint', 'type_encoded', 'season_encoded', 'is_weekend', 'month'
        ]

        first_res = df.groupby('sk_beneficiary').first().reset_index()
        first_res = first_res[first_res['is_loyal'].notna()]
        X = first_res[feature_columns].copy()
        y = first_res['is_loyal'].copy()

        Q1, Q3 = X['price'].quantile(0.25), X['price'].quantile(0.75)
        IQR    = Q3 - Q1
        mask   = (X['price'] >= Q1 - 3*IQR) & (X['price'] <= Q3 + 3*IQR)
        X, y   = X[mask], y[mask]

        scaler_classif = StandardScaler()
        X_scaled       = scaler_classif.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        k_neighbors      = min(3, max(1, int(y_train.sum()) - 1))
        smote            = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        with mlflow.start_run(run_name="train_fidelisation"):
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("months_window", 6)
            mlflow.log_param("n_samples", int(X.shape[0]))

            # LR
            lr_model = LogisticRegression(random_state=42, max_iter=1000, C=1)
            lr_model.fit(X_train_res, y_train_res)
            y_pred_lr  = lr_model.predict(X_test)
            y_proba_lr = lr_model.predict_proba(X_test)[:, 1]
            mlflow.log_metric("lr_recall",   round(float(recall_score(y_test, y_pred_lr)), 4))
            mlflow.log_metric("lr_roc_auc",  round(float(roc_auc_score(y_test, y_proba_lr)), 4))
            mlflow.sklearn.log_model(lr_model, "lr_model")

            # RF
            rf_classif = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_classif.fit(X_train_res, y_train_res)
            y_pred_rf  = rf_classif.predict(X_test)
            y_proba_rf = rf_classif.predict_proba(X_test)[:, 1]
            mlflow.log_metric("rf_recall",   round(float(recall_score(y_test, y_pred_rf)), 4))
            mlflow.log_metric("rf_roc_auc",  round(float(roc_auc_score(y_test, y_proba_rf)), 4))
            mlflow.sklearn.log_model(rf_classif, "rf_classif_model")

            # XGB
            xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
            xgb_model.fit(X_train_res, y_train_res)
            y_pred_xgb  = xgb_model.predict(X_test)
            y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
            mlflow.log_metric("xgb_recall",  round(float(recall_score(y_test, y_pred_xgb)), 4))
            mlflow.log_metric("xgb_roc_auc", round(float(roc_auc_score(y_test, y_proba_xgb)), 4))
            mlflow.sklearn.log_model(xgb_model, "xgb_model")

        # ✅ Sauvegarder avec noms uniques
        joblib.dump(lr_model,        "lr_model.pkl")
        joblib.dump(rf_classif,      "rf_classif_model.pkl")  # ✅ pas rf_model.pkl
        joblib.dump(xgb_model,       "xgb_model.pkl")
        joblib.dump(scaler_classif,  "scaler_classif.pkl")
        joblib.dump(le_type,         "le_type.pkl")
        joblib.dump(le_season,       "le_season.pkl")
        joblib.dump(feature_columns, "feature_columns.pkl")

        return {
            "status":        "success",
            "model":         "classification_fidelisation",
            "n_samples":     int(X.shape[0]),
            "taux_fidelite": f"{float(y.mean())*100:.2f}%",
            "lr_recall":     round(float(recall_score(y_test, y_pred_lr)),  4),
            "rf_recall":     round(float(recall_score(y_test, y_pred_rf)),  4),
            "xgb_recall":    round(float(recall_score(y_test, y_pred_xgb)), 4)
        }

    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail={
            "message": "Erreur training fidélisation",
            "traceback": traceback.format_exc()
        })


# =============================
# PREDICT — Régression final_price
# =============================
@app.post("/predict/price")
def predict_price(data: dict):
    try:
        load_regression_model()

        with open("columns.pkl", "rb") as f:
            trained_columns = pickle.load(f)

        required_fields = ["price", "budget", "marketing_spend",
                           "new_beneficiaries", "reservations",
                           "nb_events", "avg_spent_user"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise HTTPException(status_code=422, detail=f"Champs manquants : {missing}")

        for field in required_fields:
            if not isinstance(data[field], (int, float)):
                raise HTTPException(status_code=422, detail=f"{field} doit être un nombre")

        if data["price"] < 0 or data["budget"] < 0:
            raise HTTPException(status_code=422, detail="price et budget doivent être positifs")

        df = pd.DataFrame([data])
        if "type" in data:
            df[f"type_{data['type']}"] = 1
        if "status" in data:
            df[f"status_{data['status']}"] = 1
        df["price_budget_ratio"] = df["price"] / (df["budget"] + 1)

        for col in trained_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[trained_columns]

        pred = rf_regression_model.predict(rf_regression_scaler.transform(df))

        return {
            "status":         "success",
            "model":          "regression_final_price",
            "prediction":     float(pred[0]),
            "input_received": data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")


# =============================
# PREDICT — Fidélisation
# =============================
@app.post("/predict/fidelisation")
def predict_fidelisation(input_data: InputFidelisation):
    try:
        if lr_model is None:
            raise HTTPException(
                status_code=404,
                detail="Modèle fidélisation non chargé. Lance /train/fidelisation d'abord."
            )

        X = np.array([[
            input_data.price, input_data.budget, input_data.final_price,
            input_data.rating, input_data.visitors, input_data.marketing_spend,
            input_data.price_budget_ratio, input_data.margin,
            input_data.has_complaint, input_data.type_encoded,
            input_data.season_encoded, input_data.is_weekend, input_data.month
        ]])

        X_scaled = scaler_classif.transform(X)
        proba    = float(lr_model.predict_proba(X_scaled)[0][1])
        pred     = int(lr_model.predict(X_scaled)[0])

        if proba >= 0.9:   niveau = "Très élevé"
        elif proba >= 0.7: niveau = "Élevé"
        elif proba >= 0.5: niveau = "Moyen"
        else:              niveau = "Faible"

        action = (
            "ACTION PRIORITAIRE : Appel commercial + Offre personnalisée" if pred == 1 and proba >= 0.7
            else "ACTION STANDARD : Email de bienvenue + Programme de fidélité" if pred == 1
            else "ACTION LÉGÈRE : Newsletter standard"
        )

        return {
            "status":                "success",
            "model":                 "classification_fidelisation",
            "prediction":            pred,
            "label":                 "Fidèle" if pred == 1 else "Non fidèle",
            "probabilite_fidelite":  round(proba, 4),
            "niveau_confiance":      niveau,
            "action_recommandee":    action
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")


# =============================
# CATEGORIES
# =============================
@app.get("/categories")
def get_categories():
    try:
        # Filter categories that have at least 10 reservations in total to ensure enough data for forecasting
        query = '''
            SELECT c.category_name 
            FROM public."DIM_category" c
            JOIN fact_suivi_event fs ON c.category_id = fs.category_id
            GROUP BY c.category_name
            HAVING SUM(fs.reservations) >= 10
        '''
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df['category_name'].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================
# PREDICT â€” Forecasting
# =============================
@app.post("/predict/forecast")
def predict_forecast(data: InputForecast):
    try:
        # 1. Charger les donnÃ©es
        query = f"""
        WITH dts AS (SELECT date::timestamp AS ds, date_id FROM dim_date)
        SELECT d.ds, COALESCE(SUM(fs.reservations),0) AS y
        FROM dts d
        LEFT JOIN fact_suivi_event fs ON fs.reservation_date_fk = d.date_id
        LEFT JOIN public."DIM_category" c ON fs.category_id = c.category_id
        WHERE c.category_name = '{data.category_name}'
        GROUP BY d.ds
        ORDER BY d.ds
        """
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        if df.empty or df['y'].sum() == 0:
            raise HTTPException(status_code=404, detail=f"Not enough historical data for {data.category_name}. Minimum activity required.")

        df['ds'] = pd.to_datetime(df['ds'])
        df = df.set_index('ds').resample('MS').sum()

        # 2. EntraÃ®nement Holt-Winters
        model = ExponentialSmoothing(
            df['y'], 
            trend='add', 
            seasonal='add' if len(df) >= 24 else None, 
            seasonal_periods=12
        ).fit()

        # 3. Forecast
        forecast = model.forecast(data.horizon)
        
        results = []
        last_date = df.index[-1]
        for i, val in enumerate(forecast):
            next_date = last_date + pd.DateOffset(months=i+1)
            results.append({
                "date": next_date.strftime("%Y-%m-%d"),
                "value": round(float(max(0, val)), 2)
            })

        return {
            "status": "success",
            "category": data.category_name,
            "history": [{"date": d.strftime("%Y-%m-%d"), "value": float(v)} for d, v in df.tail(12).itertuples()],
            "forecast": results
        }

    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
