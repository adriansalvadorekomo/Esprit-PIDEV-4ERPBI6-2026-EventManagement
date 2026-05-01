import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. CONFIGURATION ET CONNEXION
# ============================================================
def get_engine():
    from settings import DATABASE_URL
    return create_engine(DATABASE_URL)

# ============================================================
# 2. LOGIQUE DE CLUSTERING (Double Entraînement + MLflow)
# ============================================================
def run_clustering_pipeline(engine, n_clusters=3, eps=2.0, min_samples=14):
    query = """
    SELECT f.price, f.budget, f.final_price, f.rating, f.visitors,
           e.type AS event_type, r.status AS reservation_status,
           c.subject AS complaint_subject, c.status AS complaint_status
    FROM fact_suivi_event f
    LEFT JOIN dim_event e ON f.event_sk = e.event_sk
    LEFT JOIN dim_reservation r ON f.reservation_sk = r.reservation_sk
    LEFT JOIN dim_complaint c ON f.id_complaint = c.id_complaint
    """
    df_raw = pd.read_sql(text(query), engine)
    
    # --- Préparation des données ---
    df_proc = df_raw.copy()
    df_proc['complaint_status_bin'] = df_proc['complaint_status'].map({'closed': 0, 'open': 1}).fillna(0)
    
    freq_map = df_proc['complaint_subject'].value_counts(normalize=True).to_dict()
    joblib.dump(freq_map, 'freq_map.joblib') 
    
    df_proc['complaint_subject_freq'] = df_proc['complaint_subject'].map(freq_map).fillna(0)
    df_model = pd.get_dummies(df_proc, columns=['event_type'], prefix='event_type')
    df_model['reservation_status'] = df_model['reservation_status'].astype(str).str.lower().str.strip()
    df_model = pd.get_dummies(df_model, columns=['reservation_status'], prefix='reservation_status')
    
    cols_to_use = [
        'budget', 'price', 'final_price', 'rating', 'visitors',
        'complaint_status_bin', 'complaint_subject_freq',
        'event_type_Corporate Event', 'event_type_Private Party',
        'event_type_Wedding', 'reservation_status_cancelled',
        'reservation_status_confirmed', 'reservation_status_pending'
    ]
    
    for col in cols_to_use:
        if col not in df_model.columns:
            df_model[col] = 0
            
    X = df_model[cols_to_use].fillna(0)
    
    # --- Transformation ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=min(7, X.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    
    # --- MLflow Clustering ---
    with mlflow.start_run(run_name="Clustering_Analysis"):
        # Log Hyperparamètres
        mlflow.log_param("km_n_clusters", n_clusters)
        mlflow.log_param("dbscan_eps", eps)
        mlflow.log_param("dbscan_min_samples", min_samples)

        # K-Means
        model_km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        km_labels = model_km.fit_predict(X_pca)
        
        # DBSCAN
        model_db = DBSCAN(eps=eps, min_samples=min_samples)
        db_labels = model_db.fit_predict(X_pca)

        # Métriques pour K-Means
        sil_km = silhouette_score(X_pca, km_labels)
        db_score_km = davies_bouldin_score(X_pca, km_labels)
        
        mlflow.log_metric("kmeans_silhouette", sil_km)
        mlflow.log_metric("kmeans_davies_bouldin", db_score_km)

        # Métriques pour DBSCAN (Attention : le score n'est calculable que s'il y a > 1 cluster hors bruit)
        n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        if n_clusters_db > 1:
            sil_db = silhouette_score(X_pca, db_labels)
            db_score_db = davies_bouldin_score(X_pca, db_labels)
            mlflow.log_metric("dbscan_silhouette", sil_db)
            mlflow.log_metric("dbscan_davies_bouldin", db_score_db)
        
        mlflow.log_metric("dbscan_n_clusters", n_clusters_db)
        
        # Log du modèle K-Means dans MLflow
        mlflow.sklearn.log_model(model_km, "kmeans_model")
        
        print(f"📊 K-Means: Silhouette={sil_km:.3f}, Davies-Bouldin={db_score_km:.3f}")

    return model_km, model_db, scaler, pca, df_raw

# ============================================================
# 3. LOGIQUE DE FIDÉLITÉ (Classification + MLflow)
# ============================================================
def get_season(month):
    if month in [12, 1, 2]: return 'hiver'
    elif month in [3, 4, 5]: return 'printemps'
    elif month in [6, 7, 8]: return 'ete'
    else: return 'automne'

def run_loyalty_classification(engine):
    query = """
    SELECT f.*, e.type, e.event_date, r.status 
    FROM fact_suivi_event f 
    LEFT JOIN dim_event e ON f.event_sk = e.event_sk 
    LEFT JOIN dim_reservation r ON f.reservation_sk = r.reservation_sk
    """
    df = pd.read_sql(text(query), engine)
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    df['month'] = df['event_date'].dt.month
    df['is_weekend'] = (df['event_date'].dt.dayofweek >= 5).astype(int)
    df['season'] = df['month'].apply(get_season)
    df['price_budget_ratio'] = df['price'] / (df['budget'] + 0.01)
    df['has_complaint'] = (~df['id_complaint'].isna()).astype(int)
    
    le_type = LabelEncoder()
    le_season = LabelEncoder()
    df['type_encoded'] = le_type.fit_transform(df['type'].fillna('inconnu'))
    df['season_encoded'] = le_season.fit_transform(df['season'])
    
    loyalty_map = {}
    for ben in df['sk_beneficiary'].unique():
        ben_data = df[df['sk_beneficiary'] == ben].sort_values('event_date')
        if len(ben_data) <= 1:
            loyalty_map[ben] = 0
        else:
            first_date = ben_data['event_date'].iloc[0]
            later = ben_data[(ben_data['event_date'] > first_date) & 
                             (ben_data['event_date'] <= first_date + pd.DateOffset(months=6))]
            loyalty_map[ben] = 1 if len(later) >= 1 else 0
    
    df['is_loyal'] = df['sk_beneficiary'].map(loyalty_map)
    
    feature_cols = ['price', 'budget', 'final_price', 'rating', 'visitors', 
                    'price_budget_ratio', 'has_complaint', 'type_encoded', 
                    'season_encoded', 'is_weekend', 'month']
    
    train_data = df.groupby('sk_beneficiary').first().reset_index()
    X = train_data[feature_cols].fillna(0)
    y = train_data['is_loyal']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run(run_name="Loyalty_RandomForest"):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Log Accuracy
        accuracy = rf.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(rf, "loyalty_model")
        
        print(f"📊 Loyalty RF: Accuracy={accuracy:.3f}")
    
    return rf, feature_cols

# ============================================================
# 4. EXECUTION PRINCIPALE
# ============================================================
if __name__ == "__main__":
    # Configuration MLflow (Optionnel : pointer vers un serveur distant)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Analyse_Evenements_BI")
    
    try:
        db_engine = get_engine()
        print("--- Démarrage de l'analyse BI avec MLflow ---")
        
        # 1. Clustering
        km_model, db_model, scaler, pca, _ = run_clustering_pipeline(
            db_engine, 
            n_clusters=3, 
            eps=2.0, 
            min_samples=14
        )
        
        cluster_names = {
            0: "Client Premium",
            1: "Client Potentiel",
            2: "Client à Risque",
            -1: "Bruit / Inclassable"
        }
        joblib.dump(cluster_names, 'cluster_names.joblib')

        # 2. Classification
        loyalty_rf, features = run_loyalty_classification(db_engine)
        
        # 3. Sauvegarde locale des artefacts (pour compatibilité API/Sheets)
        joblib.dump(km_model, 'kmeans_model.joblib')
        joblib.dump(db_model, 'dbscan_model.joblib')
        joblib.dump(loyalty_rf, 'loyalty_model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        joblib.dump(pca, 'pca.joblib')
        
        print("-" * 30)
        print("✅ Entraînement terminé et suivi dans MLflow.")
        print("Tapez 'mlflow ui' dans votre terminal pour voir les résultats.")
        print("-" * 30)

    except Exception as e:
        print(f"❌ Erreur lors de l'exécution : {e}")