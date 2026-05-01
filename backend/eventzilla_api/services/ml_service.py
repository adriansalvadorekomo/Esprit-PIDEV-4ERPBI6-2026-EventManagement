from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import subprocess

from fastapi import HTTPException
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sqlalchemy.engine import Engine
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from ..config import Settings, get_settings
from ..db import get_engine
from ..schemas.ml import (
    AnomalyRecord,
    AnomalyResponse,
    DeepLearningResponse,
    ForecastMetrics,
    ForecastResponse,
    InputFidelisation,
    InputForecast,
    InputSentiment,
    PricePredictRequest,
    RecommendationItem,
    RecommendationResponse,
    SentimentResponse,
)


POSITIVE_WORDS = [
    "excellent", "parfait", "super", "bien", "bravo", "satisfait",
    "good", "great", "perfect", "happy", "amazing", "love", "best",
]
NEGATIVE_WORDS = [
    "probleme", "mauvais", "nul", "decu", "horrible", "pire",
    "bad", "poor", "terrible", "awful", "worst", "issue", "fail",
    "complaint", "unhappy", "wrong", "broken", "delayed", "cancel",
]

DL_FEATURES = [
    "price", "budget", "final_price", "rating", "visitors",
    "marketing_spend", "price_budget_ratio", "margin", "has_complaint",
    "type_enc", "month",
]
ANOMALY_FEATURES = [
    "price", "budget", "final_price", "rating", "visitors", "marketing_spend",
]


@dataclass
class RegressionArtifacts:
    model: object
    scaler: object
    columns: list[str]
    model_mtime: float


@dataclass
class FidelisationArtifacts:
    lr_model: object
    scaler: object
    accuracy: float
    recall: float
    roc_auc: float


@dataclass
class DeepLearningArtifacts:
    model: MLPClassifier
    scaler: StandardScaler
    accuracy: float
    f1_value: float
    auc: float
    iterations: int


class MLService:
    def __init__(self, engine: Engine | None = None, settings: Settings | None = None) -> None:
        self.engine = engine or get_engine()
        self.settings = settings or get_settings()
        self._regression_artifacts: RegressionArtifacts | None = None
        self._fidelisation_artifacts: FidelisationArtifacts | None = None
        self._deep_learning_artifacts: DeepLearningArtifacts | None = None

    def _artifact(self, filename: str) -> Path:
        return self.settings.artifact_path(filename)

    def _load_pickle(self, filename: str):
        path = self._artifact(filename)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Missing artifact: {filename}")
        with path.open("rb") as handle:
            return pickle.load(handle)

    def _load_joblib(self, filename: str):
        path = self._artifact(filename)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Missing artifact: {filename}")
        return joblib.load(path)

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def train_price(self) -> dict:
        try:
            result = subprocess.run(
                ["python", "train_reg.py"],
                cwd=self.settings.backend_dir,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise HTTPException(
                status_code=500,
                detail={"message": "Price training failed", "details": exc.stderr},
            ) from exc

        self._regression_artifacts = None
        self._load_regression_artifacts()
        return {
            "status": "success",
            "model": "regression_final_price",
            "output": result.stdout,
        }

    def train_fidelisation(self) -> dict:
        script_path = self.settings.backend_dir / "app2.py"
        if not script_path.exists():
            raise HTTPException(status_code=404, detail="Legacy training script app2.py not found.")

        try:
            namespace: dict[str, object] = {}
            exec(script_path.read_text(encoding="utf-8"), namespace)
            trainer = namespace.get("train_fidelisation")
            if not callable(trainer):
                raise HTTPException(status_code=500, detail="train_fidelisation function not found.")
            result = trainer()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        self._fidelisation_artifacts = None
        return result

    def train_forecast(self) -> dict:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("forecast_holt_winters")

        query = """
        WITH dts AS (SELECT date::timestamp AS ds, date_id FROM dim_date)
        SELECT d.ds, c.category_name, COALESCE(SUM(fs.reservations), 0) AS y
        FROM dts d
        CROSS JOIN public."DIM_category" c
        LEFT JOIN fact_suivi_event fs ON fs.reservation_date_fk = d.date_id AND fs.category_id = c.category_id
        GROUP BY d.ds, c.category_name
        ORDER BY d.ds
        """
        with self.engine.connect() as connection:
            frame = pd.read_sql(query, connection)

        frame["ds"] = pd.to_datetime(frame["ds"])
        categories = frame.groupby("category_name")["y"].sum()
        categories = categories[categories > 0].index.tolist()

        results = []
        for category in categories:
            cat_data = frame[frame["category_name"] == category].copy()
            monthly = cat_data.set_index("ds").resample("MS").sum()

            if len(monthly) < 24:
                continue

            split_index = int(len(monthly) * 0.8)
            train = monthly.iloc[:split_index]
            test = monthly.iloc[split_index:]

            with mlflow.start_run(run_name=f"forecast_{category}"):
                model = ExponentialSmoothing(
                    train["y"], trend="add", seasonal="add", seasonal_periods=12
                ).fit()

                forecast = np.clip(model.forecast(len(test)).to_numpy(dtype=float), 0, None)
                actual = test["y"].to_numpy(dtype=float)
                mae = float(np.mean(np.abs(actual - forecast)))
                rmse = float(np.sqrt(np.mean((actual - forecast) ** 2)))
                denominator = np.where(actual == 0, np.nan, actual)
                mape = float(np.nanmean(np.abs((actual - forecast) / denominator)) * 100)
                if np.isnan(mape):
                    mape = 0.0

                mlflow.log_param("category", category)
                mlflow.log_param("train_points", len(train))
                mlflow.log_param("test_points", len(test))
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mape", mape)
                mlflow.sklearn.log_model(model, "model")

                results.append({"category": category, "mae": round(mae, 2), "mape": round(mape, 2)})

        return {"status": "success", "model": "Holt-Winters", "trained_categories": results}

    def _load_regression_artifacts(self) -> RegressionArtifacts:
        model_path = self._artifact("rf_model.pkl")
        scaler_path = self._artifact("scaler.pkl")
        columns_path = self._artifact("columns.pkl")
        if not model_path.exists() or not scaler_path.exists() or not columns_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Regression model artifacts not found. Run /train/price first.",
            )

        model_mtime = model_path.stat().st_mtime
        if (
            self._regression_artifacts is None
            or self._regression_artifacts.model_mtime != model_mtime
        ):
            self._regression_artifacts = RegressionArtifacts(
                model=self._load_pickle("rf_model.pkl"),
                scaler=self._load_pickle("scaler.pkl"),
                columns=self._load_pickle("columns.pkl"),
                model_mtime=model_mtime,
            )
        return self._regression_artifacts

    def _load_fidelisation_artifacts(self) -> FidelisationArtifacts:
        if self._fidelisation_artifacts is None:
            model = self._load_joblib("lr_model.pkl")
            scaler = self._load_joblib("scaler_classif.pkl")
            metrics = self._evaluate_fidelisation_model(model, scaler)
            self._fidelisation_artifacts = FidelisationArtifacts(
                lr_model=model,
                scaler=scaler,
                accuracy=metrics["accuracy"],
                recall=metrics["recall"],
                roc_auc=metrics["roc_auc"],
            )
        return self._fidelisation_artifacts

    def _evaluate_fidelisation_model(self, model: object, scaler: object) -> dict[str, float]:
        query = """
            SELECT f.sk_beneficiary, f.event_sk, f.price, f.budget, f.final_price,
                   f.rating, f.visitors, f.marketing_spend, f.id_complaint,
                   e.type, e.event_date
            FROM fact_suivi_event f
            JOIN dim_event e ON f.event_sk = e.event_sk
            WHERE f.price > 0 AND f.budget > 0
        """
        with self.engine.connect() as connection:
            frame = pd.read_sql(query, connection)

        frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
        frame = frame.sort_values(["sk_beneficiary", "event_date"])
        frame["month"] = frame["event_date"].dt.month.fillna(1).astype(int)
        frame["day_of_week"] = frame["event_date"].dt.dayofweek.fillna(0).astype(int)
        frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)

        def season_for(month: int) -> str:
            if month in (12, 1, 2):
                return "hiver"
            if month in (3, 4, 5):
                return "printemps"
            if month in (6, 7, 8):
                return "ete"
            return "automne"

        frame["season"] = frame["month"].apply(season_for)
        frame["price_budget_ratio"] = frame["price"] / (frame["budget"] + 0.01)
        frame["margin"] = frame["final_price"] - frame["price"]
        frame["has_complaint"] = (~frame["id_complaint"].isna()).astype(int)

        le_type = self._load_joblib("le_type.pkl")
        le_season = self._load_joblib("le_season.pkl")
        frame["type_encoded"] = le_type.transform(frame["type"].fillna("inconnu"))
        frame["season_encoded"] = le_season.transform(frame["season"])

        loyalty_map: dict[int, int] = {}
        for beneficiary in frame["sk_beneficiary"].dropna().unique():
            beneficiary_frame = frame[frame["sk_beneficiary"] == beneficiary].sort_values("event_date")
            if len(beneficiary_frame) <= 1:
                loyalty_map[int(beneficiary)] = 0
                continue
            first_date = beneficiary_frame["event_date"].iloc[0]
            later = beneficiary_frame[
                (beneficiary_frame["event_date"] > first_date)
                & (beneficiary_frame["event_date"] <= first_date + pd.DateOffset(months=6))
            ]
            loyalty_map[int(beneficiary)] = 1 if len(later) >= 1 else 0

        frame["is_loyal"] = frame["sk_beneficiary"].map(loyalty_map)
        first_res = frame.groupby("sk_beneficiary").first().reset_index()
        features = first_res[[
            "price", "budget", "final_price", "rating", "visitors",
            "marketing_spend", "price_budget_ratio", "margin",
            "has_complaint", "type_encoded", "season_encoded",
            "is_weekend", "month",
        ]].fillna(0)
        target = first_res["is_loyal"].fillna(0)

        if target.nunique() < 2:
            return {"accuracy": 0.0, "recall": 0.0, "roc_auc": 0.0}

        _, x_test, _, y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=42,
            stratify=target,
        )
        x_test_scaled = scaler.transform(x_test)
        y_pred = model.predict(x_test_scaled)
        y_proba = model.predict_proba(x_test_scaled)[:, 1]
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }

    def predict_price(self, data: PricePredictRequest) -> dict:
        artifacts = self._load_regression_artifacts()
        payload = data.model_dump()

        frame = pd.DataFrame([payload])
        if payload.get("type"):
            frame[f"type_{payload['type']}"] = 1
        if payload.get("status"):
            frame[f"status_{payload['status']}"] = 1
        frame["price_budget_ratio"] = frame["price"] / (frame["budget"] + 1)

        for column in artifacts.columns:
            if column not in frame.columns:
                frame[column] = 0
        frame = frame[artifacts.columns]

        prediction = artifacts.model.predict(artifacts.scaler.transform(frame))
        return {
            "status": "success",
            "model": "regression_final_price",
            "prediction": float(prediction[0]),
            "input_received": payload,
        }

    def predict_fidelisation(self, data: InputFidelisation) -> dict:
        artifacts = self._load_fidelisation_artifacts()

        frame = pd.DataFrame([{
            "price": data.price,
            "budget": data.budget,
            "final_price": data.final_price,
            "rating": data.rating,
            "visitors": data.visitors,
            "marketing_spend": data.marketing_spend,
            "price_budget_ratio": data.price_budget_ratio,
            "margin": data.margin,
            "has_complaint": data.has_complaint,
            "type_encoded": data.type_encoded,
            "season_encoded": data.season_encoded,
            "is_weekend": data.is_weekend,
            "month": data.month,
        }])
        scaled = artifacts.scaler.transform(frame)
        probability = float(artifacts.lr_model.predict_proba(scaled)[0][1])
        prediction = int(artifacts.lr_model.predict(scaled)[0])

        if probability >= 0.9:
            confidence_label = "Tres eleve"
        elif probability >= 0.7:
            confidence_label = "Eleve"
        elif probability >= 0.5:
            confidence_label = "Moyen"
        else:
            confidence_label = "Faible"

        if prediction == 1 and probability >= 0.7:
            action = "ACTION PRIORITAIRE : Appel commercial + Offre personnalisee"
        elif prediction == 1:
            action = "ACTION STANDARD : Email de bienvenue + Programme de fidelite"
        else:
            action = "ACTION LEGERE : Newsletter standard"

        return {
            "status": "success",
            "model": "classification_fidelisation",
            "prediction": prediction,
            "label": "Fidele" if prediction == 1 else "Non fidele",
            "probabilite_fidelite": round(probability, 4),
            "niveau_confiance": confidence_label,
            "action_recommandee": action,
            "accuracy": round(artifacts.accuracy, 4),
            "recall": round(artifacts.recall, 4),
            "roc_auc": round(artifacts.roc_auc, 4),
        }

    def get_categories(self) -> list[str]:
        query = """
            SELECT c.category_name
            FROM public."DIM_category" c
            JOIN fact_suivi_event fs ON c.category_id = fs.category_id
            GROUP BY c.category_name
            HAVING SUM(fs.reservations) >= 10
            ORDER BY c.category_name
        """
        with self.engine.connect() as connection:
            frame = pd.read_sql(query, connection)
        return frame["category_name"].tolist()

    def predict_forecast(self, data: InputForecast) -> ForecastResponse:
        query = """
        WITH dts AS (SELECT date::timestamp AS ds, date_id FROM dim_date)
        SELECT d.ds, COALESCE(SUM(fs.reservations), 0) AS y
        FROM dts d
        LEFT JOIN fact_suivi_event fs ON fs.reservation_date_fk = d.date_id
        LEFT JOIN public."DIM_category" c ON fs.category_id = c.category_id
        WHERE c.category_name = %(category_name)s
        GROUP BY d.ds
        ORDER BY d.ds
        """
        with self.engine.connect() as connection:
            frame = pd.read_sql(query, connection, params={"category_name": data.category_name})

        if frame.empty or float(frame["y"].sum()) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Not enough historical data for {data.category_name}.",
            )

        frame["ds"] = pd.to_datetime(frame["ds"])
        monthly = frame.set_index("ds").resample("MS").sum()

        split_index = max(12, int(len(monthly) * 0.8))
        if split_index >= len(monthly):
            split_index = len(monthly) - 1
        train = monthly.iloc[:split_index]
        test = monthly.iloc[split_index:]

        validation_model = ExponentialSmoothing(
            train["y"],
            trend="add",
            seasonal="add" if len(train) >= 24 else None,
            seasonal_periods=12,
        ).fit()
        validation_forecast = np.clip(validation_model.forecast(len(test)).to_numpy(dtype=float), 0, None)
        actual = test["y"].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(actual - validation_forecast)))
        rmse = float(np.sqrt(np.mean((actual - validation_forecast) ** 2)))
        denominator = np.where(actual == 0, np.nan, actual)
        mape = float(np.nanmean(np.abs((actual - validation_forecast) / denominator)) * 100)
        if np.isnan(mape):
            mape = 0.0

        model = ExponentialSmoothing(
            monthly["y"],
            trend="add",
            seasonal="add" if len(monthly) >= 24 else None,
            seasonal_periods=12,
        ).fit()
        forecast_values = model.forecast(data.horizon)

        history = [
            {"date": idx.strftime("%Y-%m-%d"), "value": float(value)}
            for idx, value in monthly.tail(12)["y"].items()
        ]
        forecast = []
        last_date = monthly.index[-1]
        for offset, value in enumerate(forecast_values, start=1):
            forecast.append(
                {
                    "date": (last_date + pd.DateOffset(months=offset)).strftime("%Y-%m-%d"),
                    "value": round(float(max(0, value)), 2),
                }
            )

        return ForecastResponse(
            status="success",
            category=data.category_name,
            model="Holt-Winters Exponential Smoothing",
            history=history,
            forecast=forecast,
            metrics=ForecastMetrics(
                mape=round(mape, 2),
                mae=round(mae, 2),
                rmse=round(rmse, 2),
                train_points=int(len(train)),
                test_points=int(len(test)),
            ),
        )

    def predict_sentiment(self, data: InputSentiment) -> SentimentResponse:
        text = data.text.lower()
        positive = sum(1 for word in POSITIVE_WORDS if word in text)
        negative = sum(1 for word in NEGATIVE_WORDS if word in text)

        polarity = 0.0
        if positive > negative:
            polarity = 0.5
        elif negative > positive:
            polarity = -0.5

        label = "POSITIVE" if polarity > 0 else "NEGATIVE" if polarity < 0 else "NEUTRAL"
        return SentimentResponse(
            status="success",
            text=data.text,
            polarity=polarity,
            label=label,
        )

    def recommend_events(self, beneficiary_id: int, n_reco: int = 5) -> RecommendationResponse:
        query = """
        SELECT f.sk_beneficiary, f.event_sk, f.final_price, f.rating,
               f.visitors, f.marketing_spend, f.reservations,
               e.type AS event_type, e.title AS event_title
        FROM fact_suivi_event f
        JOIN dim_event e ON f.event_sk = e.event_sk
        WHERE f.sk_beneficiary IS NOT NULL
        """
        with self.engine.connect() as connection:
            frame = pd.read_sql(query, connection)

        if frame.empty:
            raise HTTPException(status_code=404, detail="No recommendation data available.")

        encoder = LabelEncoder()
        frame["type_enc"] = encoder.fit_transform(frame["event_type"].fillna("inconnu"))

        event_features = frame.groupby("event_sk").agg(
            final_price=("final_price", "mean"),
            rating=("rating", "mean"),
            visitors=("visitors", "mean"),
            marketing_spend=("marketing_spend", "mean"),
            type_enc=("type_enc", "first"),
            event_type=("event_type", "first"),
            event_title=("event_title", "first"),
        ).fillna(0)

        scaler = StandardScaler()
        event_matrix = scaler.fit_transform(event_features[["final_price", "rating", "visitors", "marketing_spend", "type_enc"]])
        similarity = cosine_similarity(event_matrix)
        sim_df = pd.DataFrame(similarity, index=event_features.index, columns=event_features.index)

        bene_events = frame.groupby("sk_beneficiary")["event_sk"].apply(list).to_dict()
        attended = bene_events.get(beneficiary_id, [])
        all_events = event_features.index.tolist()

        def make_item(event_id: int, score: float) -> RecommendationItem:
            row = event_features.loc[event_id]
            return RecommendationItem(
                event_sk=int(event_id),
                event_type=str(row["event_type"]),
                event_title=str(row["event_title"]) if row["event_title"] else None,
                avg_rating=round(float(row["rating"]), 2),
                score=round(score, 4),
            )

        if not attended:
            top_rated = event_features.sort_values(["rating", "final_price"], ascending=False).head(n_reco)
            return RecommendationResponse(
                status="success",
                beneficiary_id=beneficiary_id,
                recommendations=[make_item(int(eid), float(event_features.loc[eid, "rating"])) for eid in top_rated.index],
                type="cold-start",
            )

        remaining = [eid for eid in all_events if eid not in attended]
        if not remaining:
            return RecommendationResponse(
                status="success", beneficiary_id=beneficiary_id, recommendations=[], type="content-based"
            )

        scores: dict[int, float] = {}
        for event_id in remaining:
            sims = [sim_df.loc[event_id, known] for known in attended if known in sim_df.index]
            scores[int(event_id)] = float(np.mean(sims)) if sims else 0.0

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_reco]
        return RecommendationResponse(
            status="success",
            beneficiary_id=beneficiary_id,
            recommendations=[make_item(eid, sc) for eid, sc in top],
            type="content-based",
        )

    def detect_anomalies(self) -> AnomalyResponse:
        query = """
        SELECT f.price, f.budget, f.final_price, f.rating,
               f.visitors, f.marketing_spend, e.type AS event_type
        FROM fact_suivi_event f
        JOIN dim_event e ON f.event_sk = e.event_sk
        WHERE f.price > 0 AND f.budget > 0 AND f.final_price > 0
        """
        with self.engine.connect() as connection:
            frame = pd.read_sql(query, connection)

        if frame.empty:
            raise HTTPException(status_code=404, detail="No anomaly data available.")

        features = frame[ANOMALY_FEATURES].fillna(0)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        contamination = 0.05
        detector = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        labels = detector.fit_predict(scaled)
        scores = detector.score_samples(scaled)

        frame = frame.copy()
        frame["anomaly"] = labels
        frame["ano_score"] = scores

        anomalies = (
            frame[frame["anomaly"] == -1]
            .sort_values("ano_score")
            .head(10)[ANOMALY_FEATURES + ["ano_score"]]
        )
        sample = [
            AnomalyRecord(
                price=float(row["price"]),
                budget=float(row["budget"]),
                final_price=float(row["final_price"]),
                rating=float(row["rating"]),
                visitors=float(row["visitors"]),
                marketing_spend=float(row["marketing_spend"]),
                ano_score=float(row["ano_score"]),
            )
            for _, row in anomalies.iterrows()
        ]

        return AnomalyResponse(
            status="success",
            total_count=int(len(frame)),
            anomaly_count=int((frame["anomaly"] == -1).sum()),
            sample_anomalies=sample,
            algorithm="Isolation Forest",
            contamination=contamination,
        )

    def _train_deep_learning_artifacts(self) -> DeepLearningArtifacts:
        query = """
        SELECT f.sk_beneficiary, f.price, f.budget, f.final_price,
               f.rating, f.visitors, f.marketing_spend, f.id_complaint,
               e.type, e.event_date
        FROM fact_suivi_event f
        JOIN dim_event e ON f.event_sk = e.event_sk
        WHERE f.price > 0 AND f.budget > 0
        """
        with self.engine.connect() as connection:
            frame = pd.read_sql(query, connection)

        if frame.empty:
            raise HTTPException(status_code=404, detail="No deep learning data available.")

        frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
        frame["month"] = frame["event_date"].dt.month.fillna(1).astype(int)
        frame["price_budget_ratio"] = frame["price"] / (frame["budget"] + 0.01)
        frame["margin"] = frame["final_price"] - frame["price"]
        frame["has_complaint"] = (~frame["id_complaint"].isna()).astype(int)
        encoder = LabelEncoder()
        frame["type_enc"] = encoder.fit_transform(frame["type"].fillna("inconnu"))

        loyalty_map: dict[int, int] = {}
        for beneficiary in frame["sk_beneficiary"].dropna().unique():
            beneficiary_frame = frame[frame["sk_beneficiary"] == beneficiary].sort_values("event_date")
            if len(beneficiary_frame) <= 1:
                loyalty_map[int(beneficiary)] = 0
                continue
            first_date = beneficiary_frame["event_date"].iloc[0]
            later = beneficiary_frame[
                (beneficiary_frame["event_date"] > first_date)
                & (beneficiary_frame["event_date"] <= first_date + pd.DateOffset(months=6))
            ]
            loyalty_map[int(beneficiary)] = 1 if len(later) >= 1 else 0

        frame["is_loyal"] = frame["sk_beneficiary"].map(loyalty_map)
        first_res = frame.groupby("sk_beneficiary").first().reset_index()
        x_values = first_res[DL_FEATURES].fillna(0)
        y_values = first_res["is_loyal"].fillna(0)

        if y_values.nunique() < 2:
            raise HTTPException(status_code=500, detail="Deep learning target has only one class.")

        x_train, x_test, y_train, y_test = train_test_split(
            x_values,
            y_values,
            test_size=0.2,
            random_state=42,
            stratify=y_values,
        )
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)
        y_proba = model.predict_proba(x_test_scaled)[:, 1]

        return DeepLearningArtifacts(
            model=model,
            scaler=scaler,
            accuracy=float(accuracy_score(y_test, y_pred)),
            f1_value=float(f1_score(y_test, y_pred, zero_division=0)),
            auc=float(roc_auc_score(y_test, y_proba)),
            iterations=int(model.n_iter_),
        )

    def predict_deep_learning(self, data: InputFidelisation) -> DeepLearningResponse:
        if self._deep_learning_artifacts is None:
            self._deep_learning_artifacts = self._train_deep_learning_artifacts()

        artifacts = self._deep_learning_artifacts
        frame = pd.DataFrame([{
            "price": data.price,
            "budget": data.budget,
            "final_price": data.final_price,
            "rating": data.rating,
            "visitors": data.visitors,
            "marketing_spend": data.marketing_spend,
            "price_budget_ratio": data.price_budget_ratio,
            "margin": data.margin,
            "has_complaint": data.has_complaint,
            "type_enc": data.type_encoded,
            "month": data.month,
        }])
        scaled = artifacts.scaler.transform(frame[DL_FEATURES])
        prediction = int(artifacts.model.predict(scaled)[0])
        confidence = float(artifacts.model.predict_proba(scaled)[0][1])

        if prediction == 1:
            note = "MLP predicts a positive loyalty outcome for this profile."
        else:
            note = "MLP predicts a negative loyalty outcome for this profile."

        return DeepLearningResponse(
            status="success",
            model="MLPClassifier(128,64,32)",
            prediction=prediction,
            confidence=confidence,
            note=note,
            accuracy=round(artifacts.accuracy, 4),
            f1_score=round(artifacts.f1_value, 4),
            auc=round(artifacts.auc, 4),
            iterations=artifacts.iterations,
        )
