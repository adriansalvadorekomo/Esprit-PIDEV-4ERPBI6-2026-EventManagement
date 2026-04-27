import pandas as pd
import psycopg2
import pickle
import os
import datetime
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ─── MLflow setup ───────────────────────────────
mlflow.set_experiment("rf_final_price")

# ─── Connexion DB ───────────────────────────────
conn = psycopg2.connect(
    dbname="DW_event",
    user="postgres",
    password="221JMT2852",
    host="localhost",
    port="5432"
)

# ─── Charger données ────────────────────────────
query = """
SELECT
    f.sk_beneficiary,
    f.event_sk,
    f.price,
    f.budget,
    f.final_price,
    f.marketing_spend,
    f.new_beneficiaries,
    f.reservations,
    e.type,
    e.event_date,
    r.status
FROM fact_suivi_event f
LEFT JOIN dim_event e ON f.event_sk = e.event_sk
LEFT JOIN dim_reservation r ON f.reservation_sk = r.reservation_sk
"""
df = pd.read_sql(query, conn)

# ─── Préparation ────────────────────────────────
df = df.fillna(0)
df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
df = df.drop(columns=['event_date'])
df = pd.get_dummies(df, columns=['type', 'status'], drop_first=True)
df['price_budget_ratio'] = df['price'] / (df['budget'] + 1)
df['nb_events'] = df.groupby('sk_beneficiary')['event_sk'].transform('count')
df['avg_spent_user'] = df.groupby('sk_beneficiary')['final_price'].transform('mean')

X = df.drop(columns=['final_price', 'sk_beneficiary', 'event_sk'])
y = df['final_price']

# ─── Split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─── Scaling ────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ─── Paramètres du modèle ───────────────────────
params = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": 42
}

# ─── MLflow Run ─────────────────────────────────
with mlflow.start_run():

    # Train
    model = RandomForestRegressor(**params)
    model.fit(X_train_scaled, y_train)

    # Métriques
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Log MLflow
    mlflow.log_params(params)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, "model")

    print(f"MAE: {mae:.2f} | R2: {r2:.4f}")

# ─── Sauvegarde versionnée ──────────────────────
os.makedirs("models", exist_ok=True)
version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

pickle.dump(model,  open(f"models/rf_model_{version}.pkl", "wb"))
pickle.dump(scaler, open(f"models/scaler_{version}.pkl",   "wb"))

# Toujours garder une version "latest"
pickle.dump(model,  open("rf_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl",   "wb"))
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))

print(f"✅ Training terminé — version: {version}")