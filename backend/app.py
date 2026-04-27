from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sqlalchemy import create_engine, text

from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Initialisation des variables globales
kmeans = None
dbscan = None
rf_loyalty = None
scaler = None
pca = None
freq_map = None
cluster_names = None
cluster_metrics = None

# ============================================================
# CHARGEMENT DES MODÈLES ET TRANSFORMATEURS
# ============================================================
try:
    # Modèles de prédiction
    kmeans = joblib.load('kmeans_model.joblib')
    dbscan = joblib.load('dbscan_model.joblib')
    rf_loyalty = joblib.load('loyalty_model.joblib')
    
    # Transformateurs
    scaler = joblib.load('scaler.joblib')
    pca = joblib.load('pca.joblib')
    
    # Dictionnaires de mapping (Fréquences et Noms de Clusters)
    try:
        freq_map = joblib.load('freq_map.joblib')
    except:
        freq_map = {}
        
    try:
        cluster_names = joblib.load('cluster_names.joblib')
    except:
        # Valeurs de secours si le fichier n'est pas trouvé
        cluster_names = {0: "Client Premium", 1: "Client Potentiel", 2: "Client à Risque", -1: "Inclassable"}

    print("All models, transformers, and mappings loaded.")
except Exception as e:
    print(f"Load error: {e}")

# Fonctions utilitaires
def get_season(month):
    if month in [12, 1, 2]: return 'hiver'
    elif month in [3, 4, 5]: return 'printemps'
    elif month in [6, 7, 8]: return 'ete'
    else: return 'automne'

def get_engine():
    return create_engine("postgresql://postgres:1400@localhost:5432/DW_event")

def load_cluster_metrics():
    global cluster_metrics
    if cluster_metrics is not None:
        return cluster_metrics

    query = """
    SELECT f.price, f.budget, f.final_price, f.rating, f.visitors,
           e.type AS event_type, r.status AS reservation_status,
           c.subject AS complaint_subject, c.status AS complaint_status
    FROM fact_suivi_event f
    LEFT JOIN dim_event e ON f.event_sk = e.event_sk
    LEFT JOIN dim_reservation r ON f.reservation_sk = r.reservation_sk
    LEFT JOIN "dim_complaint" c ON f.id_complaint = c.id_complaint
    """

    engine = get_engine()
    df_raw = pd.read_sql(text(query), engine)
    df_proc = df_raw.copy()
    df_proc['complaint_status_bin'] = df_proc['complaint_status'].map({'closed': 0, 'open': 1}).fillna(0)

    current_freq_map = df_proc['complaint_subject'].value_counts(normalize=True).to_dict()
    df_proc['complaint_subject_freq'] = df_proc['complaint_subject'].map(current_freq_map).fillna(0)
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
    metrics_scaler = StandardScaler()
    X_scaled = metrics_scaler.fit_transform(X)
    metrics_pca = PCA(n_components=min(7, X.shape[1]))
    X_pca = metrics_pca.fit_transform(X_scaled)

    metrics_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    km_labels = metrics_kmeans.fit_predict(X_pca)

    metrics_dbscan = DBSCAN(eps=2.0, min_samples=14)
    db_labels = metrics_dbscan.fit_predict(X_pca)
    db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    db_noise = int((db_labels == -1).sum())

    cluster_metrics = {
        'kmeans': {
            'silhouette_score': round(float(silhouette_score(X_pca, km_labels)), 4),
            'davies_bouldin_score': round(float(davies_bouldin_score(X_pca, km_labels)), 4),
            'n_clusters_detected': int(len(set(km_labels))),
            'noise_points': 0,
        },
        'dbscan': {
            'silhouette_score': round(float(silhouette_score(X_pca, db_labels)), 4) if db_clusters > 1 else None,
            'davies_bouldin_score': round(float(davies_bouldin_score(X_pca, db_labels)), 4) if db_clusters > 1 else None,
            'n_clusters_detected': int(db_clusters),
            'noise_points': db_noise,
        }
    }
    return cluster_metrics

# ============================================================
# ROUTE FIDÉLITÉ (Random Forest)
# ============================================================
@app.route('/predict-loyalty', methods=['POST'])
def predict_loyalty():
    try:
        data = request.get_json()
        if not data: return jsonify({'error': 'JSON vide'}), 400

        price = float(data.get('price', 0))
        budget = float(data.get('budget', 0))
        event_date = pd.to_datetime(data.get('event_date', datetime.now()))
        month = event_date.month
        
        season_map = {'hiver': 0, 'printemps': 1, 'ete': 2, 'automne': 3}
        type_map = {'Corporate Event': 0, 'Private Party': 1, 'Wedding': 2, 'inconnu': 3}
        
        processed_rf = {
            'price': price,
            'budget': budget,
            'final_price': float(data.get('final_price', 0)),
            'rating': float(data.get('rating', 0)),
            'visitors': float(data.get('visitors', 0)),
            'price_budget_ratio': price / (budget + 0.01),
            'has_complaint': 1 if data.get('id_complaint') else 0,
            'type_encoded': type_map.get(data.get('event_type'), 3),
            'season_encoded': season_map.get(get_season(month), 0),
            'is_weekend': 1 if event_date.dayofweek >= 5 else 0,
            'month': month
        }
        
        df_rf = pd.DataFrame([processed_rf])
        cols_rf = ['price', 'budget', 'final_price', 'rating', 'visitors', 'price_budget_ratio', 
                   'has_complaint', 'type_encoded', 'season_encoded', 'is_weekend', 'month']
        
        prediction = rf_loyalty.predict(df_rf[cols_rf])
        prob = rf_loyalty.predict_proba(df_rf[cols_rf])[:, 1]

        return jsonify({
            'is_loyal': int(prediction[0]), 
            'probability': round(float(prob[0]), 2), 
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================
# ROUTE CLUSTERING (KMeans & DBSCAN)
# ============================================================
@app.route('/predict-cluster', methods=['POST'])
def predict_cluster():
    try:
        data_in = request.get_json()
        if not data_in: return jsonify({'error': 'JSON vide'}), 400

        # On détermine l'algorithme (par défaut dbscan si non précisé)
        # On le récupère soit de l'objet unique, soit du premier élément de la liste
        if isinstance(data_in, dict):
            items = [data_in]
            algo = data_in.get('algo', 'dbscan').lower()
        else:
            items = data_in
            algo = items[0].get('algo', 'dbscan').lower() if items else 'dbscan'

        processed_rows = []
        for item in items:
            subject = item.get('complaint_subject', 'Autre')
            subject_freq = freq_map.get(subject, 0.0) 

            processed_rows.append({
                'budget': float(item.get('budget', 0)),
                'price': float(item.get('price', 0)),
                'final_price': float(item.get('final_price', 0)),
                'rating': float(item.get('rating', 0)),
                'visitors': float(item.get('visitors', 0)),
                'complaint_status_bin': 1 if item.get('complaint_status') == 'open' else 0,
                'complaint_subject_freq': subject_freq, 
                'event_type_Corporate Event': 1 if item.get('event_type') == 'Corporate Event' else 0,
                'event_type_Private Party': 1 if item.get('event_type') == 'Private Party' else 0,
                'event_type_Wedding': 1 if item.get('event_type') == 'Wedding' else 0,
                'reservation_status_cancelled': 1 if item.get('reservation_status') == 'cancelled' else 0,
                'reservation_status_confirmed': 1 if item.get('reservation_status') == 'confirmed' else 0,
                'reservation_status_pending': 1 if item.get('reservation_status') == 'pending' else 0
            })
        
        cols_to_use = [
            'budget', 'price', 'final_price', 'rating', 'visitors',
            'complaint_status_bin', 'complaint_subject_freq',
            'event_type_Corporate Event', 'event_type_Private Party',
            'event_type_Wedding', 'reservation_status_cancelled',
            'reservation_status_confirmed', 'reservation_status_pending'
        ]
        
        df_input = pd.DataFrame(processed_rows)[cols_to_use]
        print(f"Colonnes envoyées au scaler : {df_input.columns.tolist()}")
        
        # Transformation (Stats basées sur les 428 lignes du train)
        X_scaled = scaler.transform(df_input)
        X_pca = pca.transform(X_scaled)

        # Prédiction selon l'algorithme choisi
        if algo == 'dbscan':
            samples = dbscan.components_
            labels = dbscan.labels_[dbscan.core_sample_indices_]
            nn = NearestNeighbors(n_neighbors=10).fit(samples)
            _, indices = nn.kneighbors(X_pca)
            cluster_ids = [int(labels[idx[0]]) for idx in indices]
        else:
            cluster_ids = kmeans.predict(X_pca).tolist()
        
        # --- CONSTRUCTION DE LA RÉPONSE COMPLÈTE ---
        metrics = load_cluster_metrics().get(algo, {})
        results = []
        for c_id in cluster_ids:
            results.append({
                'cluster_id': c_id, 
                'cluster_name': cluster_names.get(c_id, "Inconnu"),
                'algorithm': algo,      # Ajouté pour Google Sheets
                'status': 'success',     # Ajouté pour Google Sheets
                'silhouette_score': metrics.get('silhouette_score'),
                'davies_bouldin_score': metrics.get('davies_bouldin_score'),
                'n_clusters_detected': metrics.get('n_clusters_detected'),
                'noise_points': metrics.get('noise_points', 0),
            })
        
        # Si n8n a envoyé une liste, on renvoie la liste complète
        return jsonify(results if isinstance(data_in, list) else results[0])
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error',
            'algorithm': 'unknown'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
