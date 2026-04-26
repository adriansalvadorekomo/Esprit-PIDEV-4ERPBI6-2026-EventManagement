import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="EventZella BI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SIDEBAR — DB CONNECTION
# ============================================================
st.sidebar.title("⚙️ Configuration")

st.sidebar.subheader("Connexion PostgreSQL")
db_user = st.sidebar.text_input("Utilisateur", value="postgres")
db_password = st.sidebar.text_input("Mot de passe", value="1400", type="password")
db_host = st.sidebar.text_input("Hôte", value="localhost")
db_port = st.sidebar.text_input("Port", value="5432")
db_name = st.sidebar.text_input("Base de données", value="dw_event_3")


@st.cache_resource
def get_engine(user, password, host, port, database):
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}")


def try_connect():
    try:
        engine = get_engine(db_user, db_password, db_host, db_port, db_name)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        return None


# ============================================================
# NAVIGATION
# ============================================================
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "📂 Module",
    ["🏠 Accueil", "🔍 Clustering", "🎯 Classification", "📉 Régression", "📈 Forecasting",
     "🤖 NLP", "🔁 Recommandation", "🧠 Deep Learning", "🚨 Anomalies"]
)

# ============================================================
# HOME PAGE
# ============================================================
if page == "🏠 Accueil":
    st.title("📊 EventZella — Dashboard BI")
    st.markdown("""
    Bienvenue dans le **tableau de bord Business Intelligence** d'EventZella.

    ### 📦 Modules principaux
    | Module | Description |
    |--------|-------------|
    | **🔍 Clustering** | Segmentation K-Means & DBSCAN + ACP |
    | **🎯 Classification** | Prédiction de fidélité (LR, RF, XGBoost) |
    | **📉 Régression** | Prédiction du montant dépensé (LR, RF) |
    | **📈 Forecasting** | Prévision de demande (ARIMA, Holt-Winters, Prophet) |

    ### 🚀 Objectifs avancés
    | Module | Description |
    |--------|-------------|
    | **🤖 NLP** | Analyse de sentiment & fréquence des réclamations |
    | **🔁 Recommandation** | Système de recommandation d'événements |
    | **🧠 Deep Learning** | Réseau de neurones MLP (classification & régression) |
    | **🚨 Anomalies** | Détection d'anomalies (Isolation Forest, LOF) |

    ### 🏆 Bonus
    - ✅ **Déploiement web** — Ce dashboard EST l'application web déployée (Streamlit)
    - 📁 **Git** — Versionnez avec `git init && git add . && git commit -m 'init'`

    ---
    **Pour commencer**, configurez votre connexion PostgreSQL dans la barre latérale, puis sélectionnez un module.
    """)

    engine = try_connect()
    if engine:
        st.success("✅ Connexion à la base de données réussie !")
    else:
        st.warning("⚠️ Impossible de se connecter. Vérifiez vos paramètres dans la barre latérale.")

# ============================================================
# CLUSTERING PAGE
# ============================================================
elif page == "🔍 Clustering":
    st.title("🔍 Analyse de Segmentation (Clustering)")

    engine = try_connect()
    if engine is None:
        st.error("❌ Connexion à la base de données échouée. Vérifiez vos paramètres.")
        st.stop()

    # Load data
    @st.cache_data
    def load_clustering_data(_engine):
        query = """
        SELECT
            f.sk_beneficiary,
            f.event_sk,
            f.reservation_sk,
            f.id_complaint,
            f.price,
            f.budget,
            f.final_price,
            f.rating,
            f.reservations,
            f.marketing_spend,
            f.new_beneficiaries,
            f.visitors,
            e.title AS event_title,
            e.event_date,
            e.type AS event_type,
            r.status AS reservation_status,
            c.subject AS complaint_subject,
            c.status AS complaint_status
        FROM fact_suivi_event f
        LEFT JOIN dim_event e ON f.event_sk = e.event_sk
        LEFT JOIN dim_reservation r ON f.reservation_sk = r.reservation_sk
        LEFT JOIN "DIM_complaint" c ON f.id_complaint = c.id_complaint
        """
        return pd.read_sql(query, _engine)

    try:
        df_raw = load_clustering_data(engine)
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        st.stop()

    st.success(f"✅ {len(df_raw)} lignes chargées")

    # --- Data Preparation ---
    df_encoded = df_raw.copy()
    df_encoded['complaint_status_bin'] = df_encoded['complaint_status'].map({'closed': 0, 'open': 1}).fillna(0)
    freq_map = df_encoded['complaint_subject'].value_counts(normalize=True)
    df_encoded['complaint_subject_freq'] = df_encoded['complaint_subject'].map(freq_map).fillna(0)
    df_encoded = pd.get_dummies(df_encoded, columns=['event_type'], drop_first=True)
    df_encoded['reservation_status'] = df_encoded['reservation_status'].astype(str).str.lower().str.strip()
    df_encoded = pd.get_dummies(df_encoded, columns=['reservation_status'], drop_first=False)

    cols_to_drop = [
        'sk_beneficiary', 'id_complaint', 'reservation_sk', 'event_date',
        'event_sk', 'event_title', 'complaint_subject', 'complaint_status'
    ]
    X_final = df_encoded.drop(columns=cols_to_drop, errors='ignore').fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)

    # --- PCA ---
    pca_full = PCA()
    pca_components = pca_full.fit_transform(X_scaled)

    # --- Sidebar Controls ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Paramètres Clustering")
    algo = st.sidebar.selectbox("Algorithme", ["K-Means", "DBSCAN"])
    n_pca_dims = st.sidebar.slider("Dimensions PCA", 2, min(13, X_scaled.shape[1]), 7)
    X_pca = pca_components[:, :n_pca_dims]

    if algo == "K-Means":
        n_clusters = st.sidebar.slider("Nombre de clusters (K)", 2, 10, 3)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_pca)
    else:
        eps = st.sidebar.slider("Epsilon (distance)", 0.1, 5.0, 2.0, step=0.1)
        min_samples = st.sidebar.slider("Min Samples", 1, 30, 14)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_pca)

    # --- Metrics ---
    n_clusters_detected = len(set(labels)) - (1 if -1 in labels else 0)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Clusters détectés", n_clusters_detected)

    if n_clusters_detected > 1:
        mask = labels != -1 if -1 in labels else np.ones(len(labels), dtype=bool)
        if mask.sum() > n_clusters_detected:
            sil = silhouette_score(X_pca[mask], labels[mask])
            db = davies_bouldin_score(X_pca[mask], labels[mask])
            col_m2.metric("Silhouette Score", f"{sil:.4f}")
            col_m3.metric("Davies-Bouldin Index", f"{db:.4f}")

    if algo == "DBSCAN":
        n_noise = int((labels == -1).sum())
        st.info(f"Points de bruit (outliers) : **{n_noise}**")

    # --- Session state init ---
    if 'cluster_sim' not in st.session_state:
        st.session_state.cluster_sim = None

    # --- PCA Scatter ---
    st.subheader("Visualisation PCA 2D")
    df_pca = pd.DataFrame(pca_components[:, :2], columns=['PC1', 'PC2'])
    df_pca['Cluster'] = labels.astype(str)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='viridis', ax=ax, s=60, alpha=0.8)
    if algo == "K-Means":
        centroids_2d = model.cluster_centers_[:, :2]
        ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], marker='X', s=200, c='red', label='Centres', zorder=5)
    if st.session_state.cluster_sim is not None:
        _sim_pt = st.session_state.cluster_sim['pca_pt']
        _asn = st.session_state.cluster_sim['assigned']
        ax.scatter(float(_sim_pt[0, 0]), float(_sim_pt[0, 1]),
                   marker='*', s=700, c='yellow', edgecolors='black',
                   linewidths=1.5, zorder=11, label=f'⭐ Votre profil → Cluster {_asn}')
    ax.set_title(f"Segmentation {algo} — Vue PCA 2D")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

    # --- Scree Plot ---
    with st.expander("📉 Scree Plot (Valeurs Propres)"):
        eigenvalues = pca_full.explained_variance_
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', color='b')
        ax2.axhline(y=1, color='r', linestyle='--', label='Critère de Kaiser')
        ax2.set_title("Valeurs propres des composantes principales")
        ax2.set_xlabel("Composante principale")
        ax2.set_ylabel("Valeur propre")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

    # --- Elbow + Silhouette (K-Means only) ---
    if algo == "K-Means":
        with st.expander("📊 Méthode du Coude & Silhouette Score"):
            col_e1, col_e2 = st.columns(2)
            k_range = range(2, 11)
            inertias = []
            sil_scores = []
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km_labels = km.fit_predict(X_pca)
                inertias.append(km.inertia_)
                sil_scores.append(silhouette_score(X_pca, km_labels))

            with col_e1:
                fig3, ax3 = plt.subplots()
                ax3.plot(list(k_range), inertias, marker='o', linestyle='--', color='b')
                ax3.set_title("Méthode du Coude")
                ax3.set_xlabel("K")
                ax3.set_ylabel("Inertie")
                ax3.grid(True, alpha=0.3)
                st.pyplot(fig3)

            with col_e2:
                fig4, ax4 = plt.subplots()
                ax4.plot(list(k_range), sil_scores, marker='o', color='green')
                ax4.set_title("Silhouette Score vs K")
                ax4.set_xlabel("K")
                ax4.set_ylabel("Silhouette Score")
                ax4.grid(True, alpha=0.3)
                st.pyplot(fig4)

    # --- K-Distance Graph (DBSCAN only) ---
    if algo == "DBSCAN":
        with st.expander("📊 Graphe K-Distance"):
            k_val = min_samples
            neigh = NearestNeighbors(n_neighbors=k_val)
            neigh.fit(X_pca)
            distances, _ = neigh.kneighbors(X_pca)
            distances_k = np.sort(distances[:, k_val - 1])
            fig5, ax5 = plt.subplots(figsize=(10, 5))
            ax5.plot(distances_k)
            ax5.set_xlabel("Points triés par distance")
            ax5.set_ylabel(f"Distance au {k_val}-ième voisin")
            ax5.set_title(f"Graphe K-distance (MinPts={k_val})")
            ax5.grid(True, alpha=0.5)
            st.pyplot(fig5)

    # --- Cluster Profiles ---
    st.subheader("Profil des Clusters")
    df_raw_copy = df_raw.copy()
    df_raw_copy['Cluster'] = labels

    numeric_cols = ['budget', 'price', 'final_price', 'rating', 'visitors']
    available_cols = [c for c in numeric_cols if c in df_raw_copy.columns]

    if n_clusters_detected > 0:
        profile = df_raw_copy[df_raw_copy['Cluster'] != -1].groupby('Cluster')[available_cols].mean()
        st.dataframe(profile.style.format("{:.2f}"), use_container_width=True)

        fig6, ax6 = plt.subplots(figsize=(12, 5))
        sns.heatmap(profile, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax6)
        ax6.set_title("Moyennes par Cluster")
        st.pyplot(fig6)

    # --- Distribution by event type ---
    if 'event_type' in df_raw_copy.columns:
        st.subheader("Répartition des Clusters par Type d'Événement")
        fig7, ax7 = plt.subplots(figsize=(10, 5))
        sns.countplot(data=df_raw_copy[df_raw_copy['Cluster'] != -1], x='event_type', hue='Cluster', ax=ax7)
        plt.xticks(rotation=45)
        ax7.set_title("Distribution Cluster × Type d'Événement")
        st.pyplot(fig7)

    # --- Raw data preview ---
    with st.expander("🗂️ Aperçu des données segmentées"):
        st.dataframe(df_raw_copy.head(30), use_container_width=True)

    # --- Cluster Simulator ---
    st.subheader("🔮 Simulateur — Assigner un Nouveau Profil")
    st.caption("Entrez les caractéristiques d'un nouvel événement pour voir à quel cluster il appartient.")

    # Collect all unique event types and reservation statuses from data
    event_type_options = sorted(df_raw['event_type'].dropna().unique().tolist()) if 'event_type' in df_raw.columns else ['Birthday']
    res_status_options = sorted(df_raw['reservation_status'].dropna().str.lower().str.strip().unique().tolist()) if 'reservation_status' in df_raw.columns else ['confirmed']

    with st.form("cluster_sim_form"):
        c1, c2, c3 = st.columns(3)
        sim_price    = c1.number_input("Prix (€)", min_value=0, value=3000, step=100)
        sim_budget   = c2.number_input("Budget (€)", min_value=0, value=20000, step=500)
        sim_final    = c3.number_input("Prix final payé (€)", min_value=0, value=5000, step=100)
        c4, c5, c6 = st.columns(3)
        sim_rating   = c4.slider("Note (rating)", 1.0, 5.0, 3.5, 0.1)
        sim_visitors = c5.number_input("Visiteurs", min_value=0, value=500, step=50)
        sim_mktg     = c6.number_input("Marketing spend (€)", min_value=0, value=10000, step=500)
        c7, c8, c9 = st.columns(3)
        sim_event_type   = c7.selectbox("Type d'événement", event_type_options)
        sim_res_status   = c8.selectbox("Statut réservation", res_status_options)
        sim_complaint_cl = c9.selectbox("Réclamation ?", ["Non (closed)", "Oui (open)"])
        submitted_cl = st.form_submit_button("🔍 Trouver le cluster")

    if submitted_cl:
        _cl_rerun = False
        try:
            sim_row = pd.DataFrame([{col: 0.0 for col in X_final.columns}])
            for col_name, val in [('price', sim_price), ('budget', sim_budget),
                                   ('final_price', sim_final), ('rating', sim_rating),
                                   ('visitors', sim_visitors), ('marketing_spend', sim_mktg)]:
                if col_name in sim_row.columns:
                    sim_row[col_name] = float(val)
            et_col = f'event_type_{sim_event_type}'
            if et_col in sim_row.columns:
                sim_row[et_col] = 1.0
            rs_col = f'reservation_status_{sim_res_status}'
            if rs_col in sim_row.columns:
                sim_row[rs_col] = 1.0
            sim_row['complaint_status_bin'] = 1.0 if "open" in sim_complaint_cl else 0.0
            sim_scaled_cl = scaler.transform(sim_row.astype(float))
            sim_pca_cl = pca_full.transform(sim_scaled_cl)[:, :n_pca_dims]

            if algo == "K-Means":
                assigned = int(model.predict(sim_pca_cl)[0])
                dists = np.linalg.norm(model.cluster_centers_ - sim_pca_cl, axis=1)
                dist_df = pd.DataFrame({'Cluster': range(len(dists)), 'Distance': dists}).sort_values('Distance')
                st.session_state.cluster_sim = {
                    'pca_pt': sim_pca_cl, 'assigned': assigned,
                    'dist_df': dist_df, 'algo': 'kmeans'
                }
            else:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(X_pca)
                idx_nn = nn.kneighbors(sim_pca_cl, return_distance=False)[0][0]
                assigned = int(labels[idx_nn])
                st.session_state.cluster_sim = {
                    'pca_pt': sim_pca_cl, 'assigned': assigned,
                    'dist_df': None, 'algo': 'dbscan'
                }
            _cl_rerun = True
        except Exception as ex:
            st.error(f"Erreur simulateur : {ex}")
        if _cl_rerun:
            st.rerun()

    # --- Cluster result (persisted via session_state) ---
    if st.session_state.get('cluster_sim') is not None:
        _r = st.session_state.cluster_sim
        _asn_r = _r['assigned']
        st.success(f"✅ Ce profil appartient au **Cluster {_asn_r}**")
        if n_clusters_detected > 0 and len(profile) > 0 and _asn_r in profile.index:
            st.dataframe(profile.loc[[_asn_r]].style.format("{:.2f}"), use_container_width=True)
        if _r['dist_df'] is not None:
            with st.expander("📏 Distances aux centroïdes — graphe"):
                _dd = _r['dist_df']
                fig_dd, ax_dd = plt.subplots(figsize=(7, 4))
                colors_dd = ['#27ae60' if int(c) == _asn_r else '#3498db' for c in _dd['Cluster']]
                ax_dd.barh(_dd['Cluster'].astype(str), _dd['Distance'], color=colors_dd)
                ax_dd.set_xlabel('Distance au centroïde')
                ax_dd.set_title('Distance de votre profil à chaque cluster')
                for i, (c, d) in enumerate(zip(_dd['Cluster'], _dd['Distance'])):
                    ax_dd.text(d + 0.01, i, f'{d:.3f}', va='center', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig_dd)

# ============================================================
# CLASSIFICATION PAGE
# ============================================================
elif page == "🎯 Classification":
    st.title("🎯 Classification — Prédiction de Fidélité")

    engine = try_connect()
    if engine is None:
        st.error("❌ Connexion à la base de données échouée.")
        st.stop()

    # --- Load Data ---
    @st.cache_data
    def load_classification_data(_engine):
        query = """
        SELECT
            f.sk_beneficiary,
            f.event_sk,
            f.price,
            f.budget,
            f.final_price,
            f.rating,
            f.visitors,
            f.marketing_spend,
            f.new_beneficiaries,
            f.id_complaint,
            f.reservations,
            e.type,
            e.event_date,
            r.status,
            d.quarter,
            d.year
        FROM fact_suivi_event f
        LEFT JOIN dim_event e ON f.event_sk = e.event_sk
        LEFT JOIN dim_reservation r ON f.reservation_sk = r.reservation_sk
        LEFT JOIN dim_date d ON f.date_event_fk = d.date_id
        """
        df = pd.read_sql(query, _engine)
        df['event_date'] = pd.to_datetime(df['event_date'])
        df = df.sort_values(['sk_beneficiary', 'event_date'])
        return df

    try:
        df = load_classification_data(engine)
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        st.stop()

    st.success(f"✅ {len(df)} lignes chargées")

    # --- Feature Engineering ---
    df['status'] = df['status'].replace({'cancellé': 'cancelled'})
    df['month'] = df['event_date'].dt.month
    df['day_of_week'] = df['event_date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    def get_season(month):
        if month in [12, 1, 2]:
            return 'hiver'
        elif month in [3, 4, 5]:
            return 'printemps'
        elif month in [6, 7, 8]:
            return 'ete'
        else:
            return 'automne'

    df['season'] = df['month'].apply(get_season)
    df['price_budget_ratio'] = df['price'] / (df['budget'] + 0.01)
    df['margin'] = df['final_price'] - df['price']
    df['has_complaint'] = (~df['id_complaint'].isna()).astype(int)

    le_type = LabelEncoder()
    le_season = LabelEncoder()
    df['type_encoded'] = le_type.fit_transform(df['type'].fillna('inconnu'))
    df['season_encoded'] = le_season.fit_transform(df['season'])

    # --- Sidebar Parameters ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Paramètres Classification")
    months_window = st.sidebar.slider("Fenêtre de fidélité (mois)", 3, 12, 6)
    test_size = st.sidebar.slider("Taille du jeu de test (%)", 10, 40, 20) / 100
    use_smote = st.sidebar.checkbox("Utiliser SMOTE (rééquilibrage)", value=True)
    remove_outliers = st.sidebar.checkbox("Supprimer les outliers (prix)", value=True)

    # --- Target: Time-window loyalty ---
    def create_loyalty_target(data, window_months=6):
        loyalty_map = {}
        for beneficiary in data['sk_beneficiary'].unique():
            ben_data = data[data['sk_beneficiary'] == beneficiary].sort_values('event_date')
            if len(ben_data) <= 1:
                loyalty_map[beneficiary] = 0
                continue
            first_date = ben_data['event_date'].iloc[0]
            later = ben_data[
                (ben_data['event_date'] > first_date) &
                (ben_data['event_date'] <= first_date + pd.DateOffset(months=window_months))
            ]
            loyalty_map[beneficiary] = 1 if len(later) >= 1 else 0
        data['is_loyal'] = data['sk_beneficiary'].map(loyalty_map)
        return data

    df = create_loyalty_target(df, window_months=months_window)

    # --- Features: first reservation per beneficiary ---
    feature_columns = [
        'price', 'budget', 'final_price', 'rating', 'visitors',
        'marketing_spend', 'price_budget_ratio', 'margin',
        'has_complaint', 'type_encoded', 'season_encoded',
        'is_weekend', 'month'
    ]

    first_reservations = df.groupby('sk_beneficiary').first().reset_index()
    first_reservations = first_reservations[first_reservations['is_loyal'].notna()]

    X = first_reservations[feature_columns].copy()
    y = first_reservations['is_loyal'].copy()

    # --- Outlier removal ---
    if remove_outliers:
        Q1, Q3 = X['price'].quantile(0.25), X['price'].quantile(0.75)
        IQR = Q3 - Q1
        mask = (X['price'] >= Q1 - 3 * IQR) & (X['price'] <= Q3 + 3 * IQR)
        X = X[mask]
        y = y[mask]

    # --- Metrics display ---
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Bénéficiaires", len(X))
    col_s2.metric("Taux de fidélité", f"{y.mean() * 100:.2f}%")
    col_s3.metric("Fidèles / Non-fidèles", f"{int(y.sum())} / {int(len(y) - y.sum())}")

    if y.nunique() < 2:
        st.error("⚠️ Une seule classe détectée. Ajustez la **fenêtre de fidélité**.")
        st.stop()

    # --- Scaling ---
    scaler_cls = StandardScaler()
    X_scaled = pd.DataFrame(scaler_cls.fit_transform(X), columns=X.columns, index=X.index)

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )
    first_res_test = first_reservations.loc[X_test.index]

    # --- SMOTE ---
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            n_fideles_train = int(y_train.sum())
            k_neighbors = min(3, n_fideles_train - 1) if n_fideles_train > 1 else 1
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            st.caption(f"SMOTE appliqué (k_neighbors={k_neighbors}) : {len(X_train_res)} échantillons d'entraînement")
        except ImportError:
            st.warning("⚠️ `imbalanced-learn` non installé. SMOTE désactivé.")
            X_train_res, y_train_res = X_train, y_train
    else:
        X_train_res, y_train_res = X_train, y_train

    # --- Models with hyperparameter tuning ---
    st.subheader("Entraînement des Modèles")

    with st.spinner("Entraînement en cours (avec tuning des hyperparamètres)..."):
        # --- Logistic Regression (tune C) ---
        best_c, best_score_lr = 0.1, 0
        for c in [0.01, 0.1, 1, 10, 100]:
            _lr = LogisticRegression(random_state=42, max_iter=1000, C=c)
            _lr.fit(X_train_res, y_train_res)
            _score = recall_score(y_test, _lr.predict(X_test), zero_division=0)
            if _score > best_score_lr:
                best_score_lr = _score
                best_c = c

        lr_model = LogisticRegression(random_state=42, max_iter=1000, C=best_c)
        lr_model.fit(X_train_res, y_train_res)
        y_pred_lr = lr_model.predict(X_test)
        y_proba_lr = lr_model.predict_proba(X_test)[:, 1]

        # --- Random Forest (tune max_depth) ---
        best_depth_rf, best_score_rf = 3, 0
        for depth in [3, 5, 7, 10, None]:
            _rf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
            _rf.fit(X_train_res, y_train_res)
            _score = recall_score(y_test, _rf.predict(X_test), zero_division=0)
            if _score > best_score_rf:
                best_score_rf = _score
                best_depth_rf = depth

        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=best_depth_rf,
            min_samples_split=5, random_state=42
        )
        rf_model.fit(X_train_res, y_train_res)
        y_pred_rf = rf_model.predict(X_test)
        y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

        # --- XGBoost (tune max_depth) ---
        xgb_available = False
        try:
            from xgboost import XGBClassifier
            best_depth_xgb, best_score_xgb = 3, 0
            for depth in [3, 5, 7, 10]:
                _xgb = XGBClassifier(n_estimators=100, max_depth=depth, learning_rate=0.1,
                                     random_state=42, use_label_encoder=False, eval_metric='logloss')
                _xgb.fit(X_train_res, y_train_res)
                _score = recall_score(y_test, _xgb.predict(X_test), zero_division=0)
                if _score > best_score_xgb:
                    best_score_xgb = _score
                    best_depth_xgb = depth

            xgb_model = XGBClassifier(n_estimators=100, max_depth=best_depth_xgb, learning_rate=0.1,
                                      random_state=42, use_label_encoder=False, eval_metric='logloss')
            xgb_model.fit(X_train_res, y_train_res)
            y_pred_xgb = xgb_model.predict(X_test)
            y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
            xgb_available = True
        except ImportError:
            st.warning("⚠️ XGBoost non installé.")

    # --- Tuning Summary ---
    with st.expander("🔧 Résultats du tuning"):
        st.write(f"- **Logistic Regression** : C={best_c} (Recall={best_score_lr:.4f})")
        st.write(f"- **Random Forest** : max_depth={best_depth_rf} (Recall={best_score_rf:.4f})")
        if xgb_available:
            st.write(f"- **XGBoost** : max_depth={best_depth_xgb} (Recall={best_score_xgb:.4f})")

    # --- Results Table ---
    results_data = {
        'Modèle': ['Régression Logistique', 'Random Forest'],
        'Accuracy': [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_rf)],
        'Precision': [precision_score(y_test, y_pred_lr, zero_division=0),
                      precision_score(y_test, y_pred_rf, zero_division=0)],
        'Recall': [recall_score(y_test, y_pred_lr, zero_division=0),
                   recall_score(y_test, y_pred_rf, zero_division=0)],
        'F1-Score': [f1_score(y_test, y_pred_lr, zero_division=0),
                     f1_score(y_test, y_pred_rf, zero_division=0)],
        'ROC-AUC': [roc_auc_score(y_test, y_proba_lr) if y_test.nunique() > 1 else 0,
                    roc_auc_score(y_test, y_proba_rf) if y_test.nunique() > 1 else 0],
    }
    if xgb_available:
        results_data['Modèle'].append('XGBoost')
        results_data['Accuracy'].append(accuracy_score(y_test, y_pred_xgb))
        results_data['Precision'].append(precision_score(y_test, y_pred_xgb, zero_division=0))
        results_data['Recall'].append(recall_score(y_test, y_pred_xgb, zero_division=0))
        results_data['F1-Score'].append(f1_score(y_test, y_pred_xgb, zero_division=0))
        results_data['ROC-AUC'].append(roc_auc_score(y_test, y_proba_xgb) if y_test.nunique() > 1 else 0)

    results_df = pd.DataFrame(results_data)

    st.subheader("📊 Comparaison des Performances")
    st.dataframe(results_df.style.format({
        'Accuracy': '{:.4f}', 'Precision': '{:.4f}',
        'Recall': '{:.4f}', 'F1-Score': '{:.4f}', 'ROC-AUC': '{:.4f}'
    }).highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                     props='background-color: #90EE90'), use_container_width=True)

    best_idx = results_df['F1-Score'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Modèle']
    st.info(f"🏆 **Meilleur modèle (F1-Score)** : {best_model_name} — "
            f"F1={results_df.loc[best_idx, 'F1-Score']:.4f}, "
            f"Recall={results_df.loc[best_idx, 'Recall']:.4f}")

    # --- Live Simulator Result (session_state) ---
    if 'cls_sim' not in st.session_state:
        st.session_state.cls_sim = None
    if st.session_state.cls_sim is not None:
        _cs = st.session_state.cls_sim
        st.subheader("🎯 Résultat du Simulateur")
        _color = "#d4edda" if _cs['pred'] == 1 else "#f8d7da"
        _label = "FIDÈLE" if _cs['pred'] == 1 else "NON FIDÈLE"
        st.markdown(f"<div style='background:{_color};padding:12px;border-radius:8px;'>"
                    f"<h3>Profil : <strong>{_label}</strong></h3></div>",
                    unsafe_allow_html=True)
        fig_proba, ax_proba = plt.subplots(figsize=(8, 1.5))
        ax_proba.barh(['Probabilité de fidélité'], [_cs['proba']], color='#27ae60' if _cs['pred'] == 1 else '#e74c3c')
        ax_proba.barh(['Probabilité de fidélité'], [1 - _cs['proba']], left=[_cs['proba']], color='#ecf0f1')
        ax_proba.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
        ax_proba.set_xlim(0, 1)
        ax_proba.set_xlabel('Probabilité')
        ax_proba.set_title(f"Indice de confiance : {_cs['proba']*100:.1f}%")
        plt.tight_layout()
        st.pyplot(fig_proba)
        st.info(_cs['action'])

    # --- ROC Curves ---
    st.subheader("Courbes ROC")
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    ax_roc.plot(fpr_lr, tpr_lr, label=f'Log. Reg. (AUC={roc_auc_score(y_test, y_proba_lr):.3f})', linewidth=2)

    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
    ax_roc.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={roc_auc_score(y_test, y_proba_rf):.3f})', linewidth=2)

    if xgb_available:
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
        ax_roc.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={roc_auc_score(y_test, y_proba_xgb):.3f})', linewidth=2)

    ax_roc.plot([0, 1], [0, 1], 'k--', label='Aléatoire', linewidth=1)
    ax_roc.set_xlabel('Taux de faux positifs (1 - Spécificité)')
    ax_roc.set_ylabel('Taux de vrais positifs (Sensibilité)')
    ax_roc.set_title('Courbes ROC — Comparaison des modèles (avec SMOTE)')
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, alpha=0.3)
    st.pyplot(fig_roc)

    # --- Confusion Matrices ---
    with st.expander("🔢 Matrices de Confusion"):
        models_for_cm = [('Régression Logistique', y_pred_lr), ('Random Forest', y_pred_rf)]
        if xgb_available:
            models_for_cm.append(('XGBoost', y_pred_xgb))
        cols_cm = st.columns(len(models_for_cm))
        for i, (name, y_pred) in enumerate(models_for_cm):
            with cols_cm[i]:
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                cmap = {'Régression Logistique': 'Blues', 'Random Forest': 'Greens', 'XGBoost': 'Oranges'}
                sns.heatmap(cm, annot=True, fmt='d', cmap=cmap.get(name, 'Blues'),
                            xticklabels=['Non fidèle', 'Fidèle'],
                            yticklabels=['Non fidèle', 'Fidèle'], ax=ax_cm)
                ax_cm.set_title(name)
                ax_cm.set_xlabel('Prédiction')
                ax_cm.set_ylabel('Réel')
                st.pyplot(fig_cm)

    # --- Feature Importance ---
    with st.expander("📊 Importance des Features (Random Forest)"):
        fi = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
        sns.barplot(data=fi.head(10), x='Importance', y='Feature', palette='viridis', ax=ax_fi)
        ax_fi.set_title("Top 10 Features les plus importantes (Random Forest)")
        st.pyplot(fig_fi)

        st.markdown("**Top 5 facteurs de fidélité :**")
        for i, (_, row) in enumerate(fi.head(5).iterrows(), 1):
            st.write(f"{i}. **{row['Feature']}** : {row['Importance']:.4f}")

    # --- Metric Comparison Bar Chart ---
    with st.expander("📊 Comparaison visuelle des métriques"):
        fig_bar, ax_bar = plt.subplots(figsize=(12, 5))
        metrics_list = ['Recall', 'Precision', 'F1-Score', 'ROC-AUC']
        x_pos = np.arange(len(metrics_list))
        width = 0.25

        values_lr_bar = [results_df.loc[0, m] for m in metrics_list]
        values_rf_bar = [results_df.loc[1, m] for m in metrics_list]
        ax_bar.bar(x_pos - width, values_lr_bar, width, label='Log. Reg.', color='#3498db')
        ax_bar.bar(x_pos, values_rf_bar, width, label='Random Forest', color='#2ecc71')

        if xgb_available:
            values_xgb_bar = [results_df.loc[2, m] for m in metrics_list]
            ax_bar.bar(x_pos + width, values_xgb_bar, width, label='XGBoost', color='#e74c3c')

        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(metrics_list)
        ax_bar.set_ylim(0, 1)
        ax_bar.legend()
        ax_bar.set_title("Comparaison des performances par métrique")

        for i, (lr_v, rf_v) in enumerate(zip(values_lr_bar, values_rf_bar)):
            ax_bar.text(i - width, lr_v + 0.02, f'{lr_v:.3f}', ha='center', fontsize=9)
            ax_bar.text(i, rf_v + 0.02, f'{rf_v:.3f}', ha='center', fontsize=9)
        if xgb_available:
            for i, xgb_v in enumerate(values_xgb_bar):
                ax_bar.text(i + width, xgb_v + 0.02, f'{xgb_v:.3f}', ha='center', fontsize=9)

        st.pyplot(fig_bar)

    # --- Predictions & Recommended Actions ---
    st.subheader("🎯 Prédictions & Actions Recommandées")

    # Use best model (LR by default as per notebook)
    best_proba = y_proba_lr
    best_pred = y_pred_lr

    predictions_df = pd.DataFrame({
        'sk_beneficiary': first_res_test['sk_beneficiary'].values,
        'probabilite_fidelite': best_proba,
        'prediction': ['Fidèle' if p == 1 else 'Non fidèle' for p in best_pred],
    })
    predictions_df['niveau_confiance'] = predictions_df['probabilite_fidelite'].apply(
        lambda p: 'Élevée' if p >= 0.7 else ('Moyenne' if p >= 0.4 else 'Faible')
    )

    def action_a_declencher(prediction, probabilite):
        if prediction == 'Fidèle':
            if probabilite >= 0.7:
                return "🔴 ACTION PRIORITAIRE : Appel commercial + Offre personnalisée"
            else:
                return "🟡 ACTION STANDARD : Email de bienvenue + Programme de fidélité"
        else:
            return "🟢 ACTION LÉGÈRE : Newsletter standard"

    predictions_df['action_recommandee'] = predictions_df.apply(
        lambda x: action_a_declencher(x['prediction'], x['probabilite_fidelite']), axis=1
    )

    st.dataframe(predictions_df.style.format({'probabilite_fidelite': '{:.4f}'}), use_container_width=True)

    # --- Action Summary ---
    st.subheader("📋 Résumé des Actions")
    action_counts = predictions_df['action_recommandee'].value_counts()
    col_a1, col_a2 = st.columns([2, 1])
    with col_a1:
        fig_act, ax_act = plt.subplots(figsize=(8, 4))
        action_counts.plot(kind='barh', ax=ax_act, color=['#2ecc71', '#f39c12', '#e74c3c'][:len(action_counts)])
        ax_act.set_xlabel("Nombre de bénéficiaires")
        ax_act.set_title("Répartition des actions recommandées")
        plt.tight_layout()
        st.pyplot(fig_act)
    with col_a2:
        for action, count in action_counts.items():
            st.write(f"**{count}** — {action}")

    # --- Scoring Simulator ---
    st.subheader("🎯 Simulateur de Scoring — Prédire la Fidélité")
    st.caption("Entrez les données d'un nouveau bénéficiaire pour prédire sa fidélité.")

    event_types_cls = sorted(df['type'].dropna().unique().tolist())
    months_fr = {
        1: 'Janvier', 2: 'Février', 3: 'Mars', 4: 'Avril',
        5: 'Mai', 6: 'Juin', 7: 'Juillet', 8: 'Août',
        9: 'Septembre', 10: 'Octobre', 11: 'Novembre', 12: 'Décembre'
    }

    with st.form("cls_sim_form"):
        col_f1, col_f2, col_f3 = st.columns(3)
        sim_budget_cls   = col_f1.number_input("Budget de l'événement (€)", min_value=0, value=1500, step=100)
        sim_price_cls    = col_f2.number_input("Prix de la réservation (€)", min_value=0, value=800, step=50)
        sim_final_cls    = col_f3.number_input("Prix final payé (€)", min_value=0, value=1400, step=50)
        col_f4, col_f5, col_f6 = st.columns(3)
        sim_visitors_cls = col_f4.slider("Nombre d'invités", 1, 500, 100)
        sim_rating_cls   = col_f5.slider("Note de satisfaction estimée", 1.0, 5.0, 4.0, 0.1)
        sim_mktg_cls     = col_f6.number_input("Marketing spend (€)", min_value=0, value=5000, step=100)
        col_f7, col_f8, col_f9 = st.columns(3)
        sim_type_cls     = col_f7.selectbox("Type d'événement", event_types_cls)
        sim_complaint    = col_f8.selectbox("Réclamation client ?", ["Non", "Oui"])
        sim_month_cls    = col_f9.selectbox("Mois de l'événement", list(months_fr.keys()), format_func=lambda m: months_fr[m])
        submitted_cls = st.form_submit_button("🔮 Calculer la probabilité de fidélité")

    if submitted_cls:
        _cls_rerun = False
        try:
            sim_dow = 0
            sim_weekend = 1 if sim_dow >= 5 else 0
            sim_season = get_season(sim_month_cls)
            sim_type_enc = le_type.transform([sim_type_cls])[0] if sim_type_cls in le_type.classes_ else 0
            sim_season_enc = le_season.transform([sim_season])[0] if sim_season in le_season.classes_ else 0
            sim_ratio = sim_price_cls / (sim_budget_cls + 0.01)
            sim_margin = sim_final_cls - sim_price_cls
            sim_has_complaint = 1 if sim_complaint == "Oui" else 0
            sim_data = pd.DataFrame([{
                'price': sim_price_cls, 'budget': sim_budget_cls, 'final_price': sim_final_cls,
                'rating': sim_rating_cls, 'visitors': sim_visitors_cls, 'marketing_spend': sim_mktg_cls,
                'price_budget_ratio': sim_ratio, 'margin': sim_margin,
                'has_complaint': sim_has_complaint, 'type_encoded': sim_type_enc,
                'season_encoded': sim_season_enc, 'is_weekend': sim_weekend, 'month': sim_month_cls
            }])
            sim_scaled_cls = scaler_cls.transform(sim_data[feature_columns])
            proba = float(lr_model.predict_proba(sim_scaled_cls)[0][1])
            pred = int(lr_model.predict(sim_scaled_cls)[0])
            action = action_a_declencher("Fidèle" if pred == 1 else "Non fidèle", proba)
            st.session_state.cls_sim = {'proba': proba, 'pred': pred, 'action': action}
            _cls_rerun = True
        except Exception as ex:
            st.error(f"Erreur simulateur : {ex}")
        if _cls_rerun:
            st.rerun()

    # --- Classification result below the form ---
    if st.session_state.get('cls_sim') is not None:
        _cs = st.session_state.cls_sim
        _color = "#d4edda" if _cs['pred'] == 1 else "#f8d7da"
        _label = "FIDÈLE" if _cs['pred'] == 1 else "NON FIDÈLE"
        st.markdown(
            f"<div style='background:{_color};padding:16px;border-radius:8px;text-align:center;'>"
            f"<h2>Résultat : <strong>{_label}</strong></h2>"
            f"<p style='font-size:18px'>Indice de confiance : <strong>{_cs['proba']*100:.1f}%</strong></p>"
            f"</div>", unsafe_allow_html=True
        )
        fig_cls_bar, ax_cls_bar = plt.subplots(figsize=(8, 1.6))
        ax_cls_bar.barh(
            ['Probabilité de fidélité'], [_cs['proba']],
            color='#27ae60' if _cs['pred'] == 1 else '#e74c3c'
        )
        ax_cls_bar.barh(
            ['Probabilité de fidélité'], [1 - _cs['proba']],
            left=[_cs['proba']], color='#ecf0f1'
        )
        ax_cls_bar.axvline(x=0.5, color='black', linestyle='--', linewidth=1.2, label='Seuil 50%')
        ax_cls_bar.set_xlim(0, 1)
        ax_cls_bar.set_xlabel('Probabilité')
        ax_cls_bar.set_title(f"Jauge de fidélité — {_cs['proba']*100:.1f}%")
        ax_cls_bar.legend(loc='lower right', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_cls_bar)
        st.info(_cs['action'])

    # --- Business Interpretation ---
    with st.expander("📊 Interprétation Business"):
        best_row = results_df.loc[best_idx]
        st.markdown(f"""
**🎯 Objectif du modèle** : Prédire si un bénéficiaire deviendra fidèle
(2+ réservations dans les {months_window} mois) à partir de sa première réservation.

**📈 Performance du meilleur modèle ({best_model_name})**
- Accuracy : **{best_row['Accuracy']:.2%}**
- Recall : **{best_row['Recall']:.2%}**
- Precision : **{best_row['Precision']:.2%}**
- F1-Score : **{best_row['F1-Score']:.4f}**
- ROC-AUC : **{best_row['ROC-AUC']:.4f}**

**💼 Recommandations**
1. **Action marketing ciblée** → Offres personnalisées aux bénéficiaires prédits "fidèles"
2. **Optimisation de l'expérience** → Renforcer les facteurs clés identifiés
3. **Programme de fidélité** → Système de points/rewards pour les fidèles
4. **Suivi & Monitoring** → Dashboard avec scores de fidélité prédits

**⚠️ Limites**
- Faible échantillon de bénéficiaires fidèles (déséquilibre extrême)
- SMOTE génère des données synthétiques pour compenser
        """)

# ============================================================
# REGRESSION PAGE
# ============================================================
elif page == "📉 Régression":
    st.title("📉 Régression — Prédiction du Montant Dépensé")

    engine = try_connect()
    if engine is None:
        st.error("❌ Connexion à la base de données échouée.")
        st.stop()

    # --- Load Data ---
    @st.cache_data
    def load_regression_data(_engine):
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
        df = pd.read_sql(query, _engine)
        return df

    try:
        df_raw_reg = load_regression_data(engine)
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        st.stop()

    st.success(f"✅ {len(df_raw_reg)} lignes chargées")

    # --- Data Preparation ---
    df_reg = df_raw_reg.copy()
    df_reg = df_reg.fillna(0)
    df_reg['event_date'] = pd.to_datetime(df_reg['event_date'], errors='coerce')
    df_reg = df_reg.drop(columns=['event_date'])

    # --- Sidebar Parameters ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Paramètres Régression")
    clip_outliers = st.sidebar.checkbox("Clipper les outliers (IQR)", value=True)
    test_size_reg = st.sidebar.slider("Taille du jeu de test (%) - Reg", 10, 40, 20) / 100
    n_features_rfe = st.sidebar.slider("Nb features RFE", 3, 10, 5)

    # --- Outlier Clipping ---
    if clip_outliers:
        for col in ['price', 'budget', 'final_price']:
            Q1 = df_reg[col].quantile(0.25)
            Q3 = df_reg[col].quantile(0.75)
            IQR = Q3 - Q1
            df_reg[col] = df_reg[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # --- Encoding ---
    df_reg = pd.get_dummies(df_reg, columns=['type', 'status'], drop_first=True)

    # --- Feature Engineering ---
    df_reg['price_budget_ratio'] = df_reg['price'] / (df_reg['budget'] + 1)
    df_reg['nb_events'] = df_reg.groupby('sk_beneficiary')['event_sk'].transform('count')
    df_reg['avg_spent_user'] = df_reg.groupby('sk_beneficiary')['final_price'].transform('mean')

    # --- Correlation Heatmap ---
    with st.expander("🔥 Matrice de Corrélation"):
        numeric_df = df_reg.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, cmap='coolwarm', ax=ax_corr, fmt='.2f',
                    linewidths=0.5, annot=True, annot_kws={'size': 7})
        ax_corr.set_title("Matrice de Corrélation")
        st.pyplot(fig_corr)

    # --- X / y ---
    X_reg = df_reg.drop(columns=['final_price', 'sk_beneficiary', 'event_sk'])
    y_reg = df_reg['final_price']

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Échantillons", len(X_reg))
    col_r2.metric("Features", X_reg.shape[1])
    col_r3.metric("Prix moyen", f"{y_reg.mean():,.0f}")

    # --- Boxplots ---
    with st.expander("📦 Boxplots (après traitement)"):
        fig_box, axes_box = plt.subplots(1, 3, figsize=(14, 4))
        for i, col in enumerate(['price', 'budget', 'final_price']):
            if col in df_reg.columns:
                axes_box[i].boxplot(df_reg[col])
                axes_box[i].set_title(f"Boxplot — {col}")
        plt.tight_layout()
        st.pyplot(fig_box)

    # --- Split & Scale ---
    from sklearn.model_selection import train_test_split, cross_val_score
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=test_size_reg, random_state=42
    )

    scaler_reg = StandardScaler()
    X_train_scaled_r = scaler_reg.fit_transform(X_train_r)
    X_test_scaled_r = scaler_reg.transform(X_test_r)

    # --- Feature Selection ---
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression, Lasso

    with st.expander("🔬 Sélection de Features (RFE & LASSO)"):
        # RFE
        rfe = RFE(estimator=LinearRegression(), n_features_to_select=n_features_rfe)
        rfe.fit(X_train_scaled_r, y_train_r)
        rfe_features = X_reg.columns[rfe.support_].tolist()
        st.write(f"**Features RFE ({n_features_rfe})** : {', '.join(rfe_features)}")

        # LASSO
        lasso = Lasso(alpha=0.01)
        lasso.fit(X_train_scaled_r, y_train_r)
        lasso_features = X_reg.columns[lasso.coef_ != 0].tolist()
        st.write(f"**Features LASSO** : {', '.join(lasso_features)}")

    # --- Train Models ---
    st.subheader("Entraînement des Modèles")

    with st.spinner("Entraînement en cours..."):
        # Linear Regression
        lr_reg = LinearRegression()
        lr_reg.fit(X_train_scaled_r, y_train_r)
        y_pred_lr_r = lr_reg.predict(X_test_scaled_r)

        # Random Forest Regressor
        from sklearn.ensemble import RandomForestRegressor
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_reg.fit(X_train_scaled_r, y_train_r)
        y_pred_rf_r = rf_reg.predict(X_test_scaled_r)

    # --- Metrics ---
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    mse_lr = mean_squared_error(y_test_r, y_pred_lr_r)
    rmse_lr = np.sqrt(mse_lr)
    mae_lr = mean_absolute_error(y_test_r, y_pred_lr_r)
    r2_lr = r2_score(y_test_r, y_pred_lr_r)

    mse_rf = mean_squared_error(y_test_r, y_pred_rf_r)
    rmse_rf = np.sqrt(mse_rf)
    mae_rf = mean_absolute_error(y_test_r, y_pred_rf_r)
    r2_rf = r2_score(y_test_r, y_pred_rf_r)

    st.subheader("📊 Comparaison des Performances")
    results_reg = pd.DataFrame({
        'Modèle': ['Linear Regression', 'Random Forest'],
        'MSE': [mse_lr, mse_rf],
        'RMSE': [rmse_lr, rmse_rf],
        'MAE': [mae_lr, mae_rf],
        'R²': [r2_lr, r2_rf]
    })
    st.dataframe(results_reg.style.format({
        'MSE': '{:,.0f}', 'RMSE': '{:,.2f}', 'MAE': '{:,.2f}', 'R²': '{:.3f}'
    }).highlight_max(axis=0, subset=['R²'], props='background-color: #90EE90')
     .highlight_min(axis=0, subset=['MSE', 'RMSE', 'MAE'], props='background-color: #90EE90'),
    use_container_width=True)

    best_model_reg = "Random Forest" if r2_rf > r2_lr else "Linear Regression"
    st.info(f"🏆 **Meilleur modèle (R²)** : {best_model_reg}")

    # --- K-Fold Cross Validation ---
    with st.expander("🔄 Validation Croisée (K-Fold, 5 folds)"):
        cv_lr = cross_val_score(lr_reg, X_train_scaled_r, y_train_r, cv=5, scoring='r2')
        cv_rf = cross_val_score(rf_reg, X_train_scaled_r, y_train_r, cv=5, scoring='r2')
        st.write(f"**Linear Regression** — R² moyen : {cv_lr.mean():.3f} ± {cv_lr.std():.3f}")
        st.write(f"**Random Forest** — R² moyen : {cv_rf.mean():.3f} ± {cv_rf.std():.3f}")

    # --- Residuals Plots ---
    st.subheader("Analyse des Résidus")
    residuals_lr_r = y_test_r - y_pred_lr_r
    residuals_rf_r = y_test_r - y_pred_rf_r

    fig_res, axes_res = plt.subplots(1, 2, figsize=(12, 5))
    axes_res[0].scatter(y_pred_lr_r, residuals_lr_r, alpha=0.5)
    axes_res[0].axhline(0, color='red', linestyle='--')
    axes_res[0].set_xlabel('Predicted')
    axes_res[0].set_ylabel('Residuals')
    axes_res[0].set_title('Résidus — Linear Regression')

    axes_res[1].scatter(y_pred_rf_r, residuals_rf_r, alpha=0.5, color='green')
    axes_res[1].axhline(0, color='red', linestyle='--')
    axes_res[1].set_xlabel('Predicted')
    axes_res[1].set_ylabel('Residuals')
    axes_res[1].set_title('Résidus — Random Forest')
    plt.tight_layout()
    st.pyplot(fig_res)

    # --- Session state init ---
    if 'reg_sim' not in st.session_state:
        st.session_state.reg_sim = None

    # --- Actual vs Predicted ---
    st.subheader("Réel vs Prédit")
    fig_avp, axes_avp = plt.subplots(1, 2, figsize=(12, 5))

    axes_avp[0].scatter(y_test_r, y_pred_lr_r, alpha=0.5)
    axes_avp[0].plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--')
    axes_avp[0].set_xlabel('Actual')
    axes_avp[0].set_ylabel('Predicted')
    axes_avp[0].set_title('Actual vs Predicted — Linear Regression')

    axes_avp[1].scatter(y_test_r, y_pred_rf_r, alpha=0.5, color='green')
    axes_avp[1].plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--')
    axes_avp[1].set_xlabel('Actual')
    axes_avp[1].set_ylabel('Predicted')
    axes_avp[1].set_title('Actual vs Predicted — Random Forest')
    plt.tight_layout()
    st.pyplot(fig_avp)

    # --- Distribution with sim prediction ---
    if st.session_state.reg_sim is not None:
        _rs = st.session_state.reg_sim
        st.subheader("📊 Distribution du Prix Final — Position de votre prédiction")
        fig_dist_r, ax_dist_r = plt.subplots(figsize=(10, 4))
        ax_dist_r.hist(y_reg, bins=40, color='#3498db', alpha=0.7, label='Distribution réelle')
        ax_dist_r.axvline(_rs['lr'], color='orange', linestyle='--', linewidth=2,
                          label=f'LR : {_rs["lr"]:,.0f} €')
        ax_dist_r.axvline(_rs['rf'], color='green', linestyle='-', linewidth=2,
                          label=f'RF : {_rs["rf"]:,.0f} €')
        ax_dist_r.axvline(y_reg.mean(), color='red', linestyle=':', linewidth=1.5,
                          label=f'Moyenne : {y_reg.mean():,.0f} €')
        ax_dist_r.set_xlabel('Prix final (€)')
        ax_dist_r.set_ylabel('Fréquence')
        ax_dist_r.set_title('Où se situe votre prédiction dans la distribution ?')
        ax_dist_r.legend()
        plt.tight_layout()
        st.pyplot(fig_dist_r)
        col_rd1, col_rd2 = st.columns(2)
        col_rd1.metric("📐 Linear Regression", f"{_rs['lr']:,.0f} €")
        col_rd2.metric("🌲 Random Forest", f"{_rs['rf']:,.0f} €")
        st.info(_rs['profile'])

    # --- Q-Q Plot & Homoscedasticity ---
    with st.expander("📐 Vérification des hypothèses (LR)"):
        import scipy.stats as stats_module
        fig_qq, axes_qq = plt.subplots(1, 2, figsize=(12, 5))
        stats_module.probplot(residuals_lr_r, dist="norm", plot=axes_qq[0])
        axes_qq[0].set_title('Q-Q Plot — Normalité des résidus (LR)')

        axes_qq[1].scatter(y_pred_lr_r, residuals_lr_r, alpha=0.5)
        axes_qq[1].axhline(0, color='red', linestyle='--')
        axes_qq[1].set_xlabel('Fitted Values')
        axes_qq[1].set_ylabel('Residuals')
        axes_qq[1].set_title('Homoscédasticité (LR)')
        plt.tight_layout()
        st.pyplot(fig_qq)

    # --- Feature Importance (RF) ---
    with st.expander("📊 Importance des Features (Random Forest)"):
        fi_reg = pd.DataFrame({
            'Feature': X_reg.columns,
            'Importance': rf_reg.feature_importances_
        }).sort_values('Importance', ascending=False)

        fig_fi_r, ax_fi_r = plt.subplots(figsize=(10, 6))
        sns.barplot(data=fi_reg.head(10), x='Importance', y='Feature', palette='viridis', ax=ax_fi_r)
        ax_fi_r.set_title("Top 10 Features — Random Forest Regressor")
        st.pyplot(fig_fi_r)

    # --- LR Coefficients ---
    with st.expander("📊 Coefficients (Linear Regression)"):
        coef_df = pd.DataFrame({
            'Feature': X_reg.columns,
            'Coefficient': lr_reg.coef_
        })
        coef_df['abs_coef'] = coef_df['Coefficient'].abs()
        coef_df = coef_df.sort_values('abs_coef', ascending=False)

        fig_coef, ax_coef = plt.subplots(figsize=(10, 6))
        colors_coef = ['#e74c3c' if c < 0 else '#2ecc71' for c in coef_df['Coefficient']]
        ax_coef.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_coef)
        ax_coef.set_xlabel('Coefficient')
        ax_coef.set_title('Coefficients — Linear Regression')
        ax_coef.invert_yaxis()
        st.pyplot(fig_coef)

        st.dataframe(coef_df[['Feature', 'Coefficient', 'abs_coef']].style.format({
            'Coefficient': '{:.2f}', 'abs_coef': '{:.2f}'
        }), use_container_width=True)

        # --- Price Prediction Simulator ---
    st.subheader("Simulateur - Predire le Montant Depense")
    st.caption("Entrez les caracteristiques d'une reservation pour estimer le prix final.")
    
    event_types_reg = ["Corporate Event", "Wedding", "Birthday", "Other"]
    status_types_reg = ["confirmed", "pending", "cancelled"]
    
    with st.form("reg_sim_form"):
        rc1, rc2, rc3 = st.columns(3)
        sim_price_r = rc1.number_input("Prix de l'evenement (EUR)", min_value=0.0, value=1500.0, step=100.0)
        sim_budget_r = rc2.number_input("Budget alloue (EUR)", min_value=0.0, value=1300.0, step=100.0)
        sim_mktg_r = rc3.number_input("Marketing spend (EUR)", min_value=0.0, value=180.0, step=50.0)
    
        rc4, rc5 = st.columns(2)
        sim_newben_r = rc4.number_input("Nouveaux beneficiaires", min_value=0, value=12, step=1)
        sim_res_r = rc5.number_input("Reservations", min_value=0, value=25, step=1)
    
        rc6, rc7 = st.columns(2)
        sim_type_r = rc6.selectbox("Type d'evenement", event_types_reg)
        sim_status_r = rc7.selectbox("Statut reservation", status_types_reg)
    
        sim_nbevents_r = st.number_input("Nb evenements (beneficiaire)", min_value=0, value=7, step=1)
        sim_avgspent_r = st.number_input("Depense moyenne utilisateur (EUR)", min_value=0.0, value=950.0, step=50.0)
    
        price_budget_ratio = sim_price_r / (sim_budget_r + 1)
        st.info(f"Price/Budget ratio calcule automatiquement : {price_budget_ratio:.2f}")
    
        submitted_reg = st.form_submit_button("Predire le prix final")
    
    if submitted_reg:
        try:
            sim_r = {col: 0.0 for col in X_reg.columns}
    
            # Variables saisies par l'utilisateur
            sim_r["price"] = float(sim_price_r)
            sim_r["budget"] = float(sim_budget_r)
            sim_r["marketing_spend"] = float(sim_mktg_r)
            sim_r["new_beneficiaries"] = float(sim_newben_r)
            sim_r["reservations"] = float(sim_res_r)
            sim_r["nb_events"] = float(sim_nbevents_r)
            sim_r["avg_spent_user"] = float(sim_avgspent_r)
    
            # Feature calculee automatiquement
            sim_r["price_budget_ratio"] = float(price_budget_ratio)
    
            # Variables dummy
            type_col = f"type_{sim_type_r}"
            if type_col in X_reg.columns:
                sim_r[type_col] = 1.0
    
            status_col = f"status_{sim_status_r}"
            if status_col in X_reg.columns:
                sim_r[status_col] = 1.0
    
            # Ordre exact attendu par le modele
            sim_df_r = pd.DataFrame([sim_r], columns=X_reg.columns).fillna(0.0).astype(float)
    
            sim_scaled_r = scaler_reg.transform(sim_df_r)
            pred_lr_r_s = float(lr_reg.predict(sim_scaled_r)[0])
            pred_rf_r_s = float(rf_reg.predict(sim_scaled_r)[0])
    
            if pred_rf_r_s > 10000:
                profile_msg = "Profil HAUT DEPENSIER"
            elif pred_rf_r_s > 4000:
                profile_msg = "Profil DEPENSE MOYENNE"
            else:
                profile_msg = "Profil FAIBLE DEPENSE"
    
            st.session_state.reg_sim = {
                "lr": pred_lr_r_s,
                "rf": pred_rf_r_s,
                "profile": profile_msg,
            }
            st.rerun()
    
        except Exception as ex:
            st.error(f"Erreur simulateur : {ex}")
    
    
        # --- Business Interpretation ---
        with st.expander("📊 Interprétation Business"):
            st.markdown(f"""
    **🎯 Objectif** : Prédire le montant dépensé (`final_price`) par un utilisateur pour optimiser le marketing et personnaliser les offres.
    
    **📈 Résultats**
    - **Linear Regression** : R² = {r2_lr:.3f}, RMSE = {rmse_lr:,.0f}
    - **Random Forest** : R² = {r2_rf:.3f}, RMSE = {rmse_rf:,.0f}
    - **Meilleur modèle** : {best_model_reg}
    
    **🔍 Facteurs clés**
    - `avg_spent_user` : Montant moyen dépensé (le plus influent)
    - `price_budget_ratio` : Ratio prix/budget
    - `nb_events` : Nombre d'événements par bénéficiaire
    
    **💼 Recommandations**
    1. **Identifier les hauts dépensiers** → Offres premium ciblées
    2. **Optimiser les budgets marketing** → Focus sur les profils à fort ROI
    3. **Personnalisation des offres** → Adapter les prix selon le profil de dépense
    
    **⚠️ Limites**
    - LR suppose une relation linéaire (hypothèse vérifiée via Q-Q plot)
    - RF plus précis mais moins interprétable (boîte noire partielle)
            """)

# ============================================================
# FORECASTING PAGE
# ============================================================
elif page == "📈 Forecasting":
    st.title("📈 Prévision  de réservations ")
    st.caption("Prévision des réservations par catégorie d'événement.")

    engine = try_connect()
    if engine is None:
        st.error("❌ Connexion à la base de données échouée.")
        st.stop()

    # ── Imports optionnels ──────────────────────────────────────
    PROPHET_OK = False
    STATS_OK   = False
    LGB_OK     = False

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        STATS_OK = True
    except ImportError:
        st.warning("⚠️ `statsmodels` non installé.")

    try:
        from prophet import Prophet
        import logging, os, sys
        from contextlib import contextmanager
        logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
        @contextmanager
        def _silence():
            devnull = open(os.devnull, 'w')
            old = sys.stderr; sys.stderr = devnull
            try: yield
            finally: sys.stderr = old; devnull.close()
        PROPHET_OK = True
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        LGB_OK = True
    except ImportError:
        pass

    # ── Métriques robustes ──────────────────────────────────────
    def _smape(a, p):
        a, p = np.array(a, float), np.array(p, float)
        d = (np.abs(a) + np.abs(p)) / 2.0
        m = d > 1e-8
        return float(np.mean(np.abs(a[m]-p[m])/d[m])*100) if m.any() else 0.0

    def _mase(a, p, train_arr):
        a, p = np.array(a, float), np.array(p, float)
        naive_mae = np.mean(np.abs(np.diff(train_arr)))
        if naive_mae < 1e-8: naive_mae = 1.0
        return float(np.mean(np.abs(a-p)) / naive_mae)

    def metrics(y_true, y_pred, y_train):
        if y_pred is None: return None, None, None, None
        a, p = np.array(y_true, float), np.array(y_pred, float)
        mae  = float(np.mean(np.abs(a-p)))
        rmse = float(np.sqrt(np.mean((a-p)**2)))
        return mae, rmse, _smape(a, p), _mase(a, p, y_train)

    # ── Données ─────────────────────────────────────────────────
    @st.cache_data
    def load_fc(_engine):
        q = """
        WITH cats AS (SELECT DISTINCT category_id, category_name FROM public."DIM_category"),
             dts  AS (SELECT date::timestamp AS ds, date_id FROM dim_date)
        SELECT d.ds, c.category_name AS category,
               COALESCE(SUM(fs.reservations),0)    AS y,
               COALESCE(SUM(fs.marketing_spend),0) AS mktg,
               COALESCE(SUM(fs.visitors),0)        AS visitors
        FROM dts d
        JOIN cats c ON TRUE
        LEFT JOIN fact_suivi_event fs
               ON fs.reservation_date_fk = d.date_id
              AND fs.category_id = c.category_id
        GROUP BY d.ds, c.category_name
        ORDER BY c.category_name, d.ds
        """
        df = pd.read_sql(q, _engine)
        df['ds'] = pd.to_datetime(df['ds'])
        return df

    try:
        raw = load_fc(engine)
    except Exception as e:
        st.error(f"Erreur chargement : {e}")
        st.stop()

    # ── Sidebar ─────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Paramètres Forecasting")

    granularity = st.sidebar.radio(
        "Granularité",
        ["Mensuelle (recommandée)", "Hebdomadaire"],
        index=0,
        help="Mensuelle réduit les zéros et stabilise les modèles."
    )
    USE_MONTHLY = "Mensuelle" in granularity

    min_act  = st.sidebar.slider("Activité minimale", 10, 300, 50)
    pct_train = st.sidebar.slider("% Données d'entraînement", 60, 90, 80) / 100

    activity   = raw.groupby('category')['y'].sum()
    valid_cats = activity[activity > min_act].index.tolist()
    if not valid_cats:
        st.warning("Aucune catégorie avec suffisamment d'activité.")
        st.stop()

    cat = st.sidebar.selectbox("Catégorie", valid_cats)

    freq = 'MS' if USE_MONTHLY else 'W'   # MS = Month Start
    label_freq = "mois" if USE_MONTHLY else "semaine"

    def prepare_series_for_category(category_name):
        df_cat_local = raw[raw['category'] == category_name].copy()
        df_ts_local = (
            df_cat_local.set_index('ds')
            .resample(freq)
            .agg({'y': 'sum', 'mktg': 'sum', 'visitors': 'sum'})
            .reset_index()
        )

        y_arr_local = df_ts_local['y'].values
        last_active_local = len(y_arr_local) - 1
        while last_active_local > 0 and y_arr_local[last_active_local] <= 0:
            last_active_local -= 1
        df_ts_local = df_ts_local.iloc[:last_active_local + 1].copy()

        zero_pct_local = (df_ts_local['y'] == 0).mean()
        if zero_pct_local < 0.40:
            df_ts_local['y'] = df_ts_local['y'].replace(0, np.nan).interpolate('linear').fillna(0)

        return df_ts_local, zero_pct_local

    def make_lags(df_in, lags):
        d = df_in.copy()
        for l in lags:
            d[f'lag_{l}'] = d['y'].shift(l)
        d['roll_3'] = d['y'].shift(1).rolling(3).mean()
        d['roll_6'] = d['y'].shift(1).rolling(6).mean()
        d['trend'] = np.arange(len(d))
        d['month'] = d['ds'].dt.month
        if USE_MONTHLY:
            d['quarter'] = d['ds'].dt.quarter
        d['mktg_l1'] = d['mktg'].shift(1)
        d['vis_l1'] = d['visitors'].shift(1)
        return d.dropna()

    def build_prophet_model(changepoint_prior_scale, yearly_s, has_mktg, has_vis):
        model = Prophet(
            growth='linear',
            yearly_seasonality=yearly_s,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=10.0,
            interval_width=0.80,
        )
        if has_mktg:
            model.add_regressor('mktg_norm')
        if has_vis:
            model.add_regressor('vis_norm')
        return model

    def prepare_prophet_frame(df_source, scale, has_mktg, has_vis, mktg_scale, vis_scale):
        df_prophet = df_source[['ds', 'y']].copy()
        df_prophet['y'] = df_prophet['y'] / scale
        if has_mktg:
            df_prophet['mktg_norm'] = df_source['mktg'].values / mktg_scale
        if has_vis:
            df_prophet['vis_norm'] = df_source['visitors'].values / vis_scale
        return df_prophet

    # ── Préparation de la série ──────────────────────────────────
    df_ts, zero_pct = prepare_series_for_category(cat)

    n_pts = len(df_ts)

    # Métriques de qualité des données
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Périodes ({label_freq}s)", n_pts)
    c2.metric("Moyenne / période", f"{df_ts['y'].mean():.0f}")
    c3.metric("Maximum", f"{df_ts['y'].max():.0f}")
    c4.metric("Zéros", f"{zero_pct:.0%}", help="Part de périodes sans activité")

    if zero_pct >= 0.40:
        st.warning(
            f"⚠️ {zero_pct:.0%} de périodes à zéro — série très creuse. "
            "Passez en granularité **mensuelle** ou choisissez une catégorie plus active."
        )

    MIN_POINTS = 12 if USE_MONTHLY else 24
    if n_pts < MIN_POINTS:
        st.error(f"Pas assez de données pour prévoir ({n_pts} {label_freq}s, minimum {MIN_POINTS}).")
        st.stop()

    # ── Train / Test ─────────────────────────────────────────────
    split = int(n_pts * pct_train)
    train = df_ts.iloc[:split].copy()
    test  = df_ts.iloc[split:].copy()

    st.write(
        f"**Entraînement** : {len(train)} {label_freq}s "
        f"({train['ds'].min().strftime('%b %Y')} → {train['ds'].max().strftime('%b %Y')})  |  "
        f"**Test** : {len(test)} {label_freq}s "
        f"({test['ds'].min().strftime('%b %Y')} → {test['ds'].max().strftime('%b %Y')})"
    )

    def train_benchmark_models(df_ts_local):
        n_pts_local = len(df_ts_local)
        split_local = int(n_pts_local * pct_train)
        train_local = df_ts_local.iloc[:split_local].copy()
        test_local = df_ts_local.iloc[split_local:].copy()
        bench_local = {}

        naive_pred = np.full(len(test_local), train_local['y'].iloc[-1])
        m_local = metrics(test_local['y'], naive_pred, train_local['y'].values)
        bench_local['Naïf'] = {
            'mae': m_local[0], 'rmse': m_local[1], 'smape': m_local[2], 'mase': m_local[3],
            'pred': naive_pred, 'ci': None
        }

        if STATS_OK:
            SARIMA_CONFIGS = [
                ((1, 1, 1), (0, 0, 0, 0)),
                ((0, 1, 1), (0, 0, 0, 0)),
                ((2, 1, 1), (0, 0, 0, 0)),
                ((1, 1, 2), (0, 0, 0, 0)),
            ]
            period = 12 if USE_MONTHLY else 52
            if len(train_local) >= 2 * period:
                SARIMA_CONFIGS += [
                    ((1, 1, 1), (1, 0, 1, period)),
                    ((1, 1, 1), (0, 1, 1, period)),
                ]

            best_aic, best_p, best_ci_s = float('inf'), None, None
            for (order_cfg, seasonal_cfg) in SARIMA_CONFIGS:
                if seasonal_cfg[3] > 0 and len(train_local) < 2 * seasonal_cfg[3]:
                    continue
                try:
                    fit = SARIMAX(
                        train_local['y'], order=order_cfg, seasonal_order=seasonal_cfg,
                        enforce_stationarity=False, enforce_invertibility=False
                    ).fit(disp=False)
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        fc_obj = fit.get_forecast(len(test_local))
                        best_p = np.clip(fc_obj.predicted_mean.values, 0, None)
                        ci_df = fc_obj.conf_int(alpha=0.20)
                        best_ci_s = (np.clip(ci_df.iloc[:, 0].values, 0, None), ci_df.iloc[:, 1].values)
                except Exception:
                    pass

            if best_p is not None:
                m_local = metrics(test_local['y'], best_p, train_local['y'].values)
                bench_local['SARIMA'] = {
                    'mae': m_local[0], 'rmse': m_local[1], 'smape': m_local[2], 'mase': m_local[3],
                    'pred': best_p, 'ci': best_ci_s
                }

        if STATS_OK:
            try:
                period_hw = 12 if USE_MONTHLY else 52
                use_seasonal = len(train_local) >= 2 * period_hw
                hw = ExponentialSmoothing(
                    train_local['y'].values,
                    trend='add',
                    seasonal='add' if use_seasonal else None,
                    seasonal_periods=period_hw if use_seasonal else None,
                    initialization_method='estimated'
                ).fit(optimized=True)
                hw_pred = np.clip(hw.forecast(len(test_local)), 0, None)
                resid_std = np.std(train_local['y'].values - hw.fittedvalues)
                hw_ci = (np.clip(hw_pred - 1.28 * resid_std, 0, None), hw_pred + 1.28 * resid_std)
                m_local = metrics(test_local['y'], hw_pred, train_local['y'].values)
                bench_local['Holt-Winters'] = {
                    'mae': m_local[0], 'rmse': m_local[1], 'smape': m_local[2], 'mase': m_local[3],
                    'pred': hw_pred, 'ci': hw_ci
                }
            except Exception:
                pass

        if PROPHET_OK:
            try:
                scale = max(train_local['y'].median(), train_local['y'].mean(), 1.0)
                has_mktg = train_local['mktg'].sum() > 0
                has_vis = train_local['visitors'].sum() > 0
                mktg_scale = max(train_local['mktg'].max(), 1e-8)
                vis_scale = max(train_local['visitors'].max(), 1e-8)
                df_p = prepare_prophet_frame(
                    train_local, scale, has_mktg, has_vis, mktg_scale, vis_scale
                )

                GRID_CPP = [0.01, 0.05, 0.15, 0.30]
                best_smape_p, best_pred_p, best_ci_p = float('inf'), None, None
                best_prophet_cfg = None
                yearly_s = n_pts_local >= (12 if USE_MONTHLY else 52)

                for cpp in GRID_CPP:
                    try:
                        m_p = build_prophet_model(cpp, yearly_s, has_mktg, has_vis)
                        with _silence():
                            m_p.fit(df_p)

                        fut = test_local[['ds']].copy()
                        if has_mktg:
                            fut['mktg_norm'] = test_local['mktg'].values / mktg_scale
                        if has_vis:
                            fut['vis_norm'] = test_local['visitors'].values / vis_scale

                        fc_p = m_p.predict(fut)
                        pred_p = np.clip(fc_p['yhat'].values * scale, 0, None)
                        sp_p = _smape(test_local['y'].values, pred_p)

                        if sp_p < best_smape_p:
                            best_smape_p = sp_p
                            best_pred_p = pred_p
                            best_ci_p = (
                                np.clip(fc_p['yhat_lower'].values * scale, 0, None),
                                fc_p['yhat_upper'].values * scale
                            )
                            best_prophet_cfg = {
                                'scale': scale,
                                'cpp': cpp,
                                'yearly_s': yearly_s,
                                'has_mktg': has_mktg,
                                'has_vis': has_vis,
                                'mktg_scale': mktg_scale,
                                'vis_scale': vis_scale,
                            }
                    except Exception:
                        pass

                if best_pred_p is not None and best_prophet_cfg is not None:
                    m_local = metrics(test_local['y'], best_pred_p, train_local['y'].values)
                    bench_local['Prophet'] = {
                        'mae': m_local[0], 'rmse': m_local[1], 'smape': m_local[2], 'mase': m_local[3],
                        'pred': best_pred_p, 'ci': best_ci_p, 'cfg': best_prophet_cfg
                    }
            except Exception:
                pass

        if LGB_OK and n_pts_local >= 30:
            try:
                lags = [1, 2, 3, 6, 12] if USE_MONTHLY else [1, 2, 4, 8, 13, 26]
                feat_cols = (
                    [f'lag_{l}' for l in lags]
                    + ['roll_3', 'roll_6', 'trend', 'month', 'mktg_l1', 'vis_l1']
                    + (['quarter'] if USE_MONTHLY else [])
                )

                df_lag = make_lags(df_ts_local, lags)
                avail = [f for f in feat_cols if f in df_lag.columns]
                split_l = int(len(df_lag) * pct_train)
                tr_l, te_l = df_lag.iloc[:split_l], df_lag.iloc[split_l:]

                if len(tr_l) >= 8 and len(te_l) >= 2:
                    lgb_m = lgb.LGBMRegressor(
                        n_estimators=400, learning_rate=0.03,
                        num_leaves=12, min_child_samples=3,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=42, verbose=-1
                    )
                    lgb_m.fit(
                        tr_l[avail], tr_l['y'],
                        eval_set=[(te_l[avail], te_l['y'])],
                        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)]
                    )
                    lgb_pred = np.clip(lgb_m.predict(te_l[avail]), 0, None)
                    n_align = min(len(lgb_pred), len(test_local['y']))
                    m_local = metrics(
                        test_local['y'].values[-n_align:], lgb_pred[-n_align:], train_local['y'].values
                    )
                    bench_local['LightGBM'] = {
                        'mae': m_local[0], 'rmse': m_local[1], 'smape': m_local[2], 'mase': m_local[3],
                        'pred': lgb_pred[-n_align:], 'ci': None, 'feat': avail, 'model': lgb_m, 'lags': lags
                    }
            except Exception:
                pass

        eligible_local = {
            n: v for n, v in bench_local.items()
            if n != 'Naïf' and v['smape'] is not None and v['smape'] > 0
        }
        if len(eligible_local) >= 2:
            try:
                weights = {n: 1.0 / v['smape'] for n, v in eligible_local.items()}
                w_total = sum(weights.values())
                min_len = min(len(v['pred']) for v in eligible_local.values())
                ens_pred = sum(
                    (weights[n] / w_total) * v['pred'][:min_len]
                    for n, v in eligible_local.items()
                )
                ens_pred = np.clip(ens_pred, 0, None)
                y_test_aligned = test_local['y'].values[:min_len]
                m_local = metrics(y_test_aligned, ens_pred, train_local['y'].values)
                bench_local['Ensemble'] = {
                    'mae': m_local[0], 'rmse': m_local[1], 'smape': m_local[2], 'mase': m_local[3],
                    'pred': ens_pred, 'ci': None
                }
            except Exception:
                pass

        benchmark_candidates = [
            name for name in bench_local
            if name not in ('Naïf', 'Ensemble') and bench_local[name]['smape'] is not None
        ]
        best_benchmark_name_local = None
        if benchmark_candidates:
            best_benchmark_name_local = min(
                benchmark_candidates,
                key=lambda name: bench_local[name]['smape']
            )

        return train_local, test_local, bench_local, eligible_local, best_benchmark_name_local

    # ── Entraînement des modèles ─────────────────────────────────
    train, test, bench, eligible, best_benchmark_name = train_benchmark_models(df_ts)

    if not bench:
        st.error("Aucun modèle n'a pu être entraîné.")
        st.stop()

    # ── Tableau Benchmark ────────────────────────────────────────
    st.subheader("📊 Comparaison des modèles")

    rows = []
    for name, v in bench.items():
        rows.append({'Modèle': name, 'MAE': v['mae'], 'RMSE': v['rmse'],
                     'SMAPE (%)': v['smape'], 'MASE': v['mase']})
    bench_df = pd.DataFrame(rows)

    # Identifier le meilleur modèle
    best_idx  = bench_df['SMAPE (%)'].idxmin()
    best_name = bench_df.loc[best_idx, 'Modèle']
    best_smape = bench_df.loc[best_idx, 'SMAPE (%)']
    best_mase  = bench_df.loc[best_idx, 'MASE']

    def style_mase(val):
        """Couleur claire et lisible pour MASE (fond clair, texte foncé)."""
        if val is None or np.isnan(val):
            return 'background-color: #f0f0f0; color: #333'
        if val < 1.0:
            return 'background-color: #c6efce; color: #276221; font-weight: bold'
        elif val < 1.3:
            return 'background-color: #ffeb9c; color: #9c5700; font-weight: bold'
        else:
            return 'background-color: #ffc7ce; color: #9c0006; font-weight: bold'

    def style_smape(val):
        if val is None or np.isnan(val):
            return 'background-color: #f0f0f0; color: #333'
        if val < 25:
            return 'background-color: #c6efce; color: #276221; font-weight: bold'
        elif val < 40:
            return 'background-color: #ffeb9c; color: #9c5700; font-weight: bold'
        else:
            return 'background-color: #ffc7ce; color: #9c0006; font-weight: bold'

    styled = (
        bench_df.style
        .format({'MAE': '{:.1f}', 'RMSE': '{:.1f}',
                 'SMAPE (%)': '{:.1f}%', 'MASE': '{:.3f}'}, na_rep='—')
        .applymap(style_mase,  subset=['MASE'])
        .applymap(style_smape, subset=['SMAPE (%)'])
        .highlight_min(axis=0, subset=['MAE','RMSE'], color='#c6efce')
        .set_properties(**{'text-align': 'center'})
    )
    st.dataframe(styled, use_container_width=True, height=min(250, 60 + 35 * len(bench_df)))

    # Légende MASE — visible, en français
    col_leg1, col_leg2, col_leg3 = st.columns(3)
    col_leg1.markdown(
        "<span style='background:#c6efce;color:#276221;padding:3px 8px;"
        "border-radius:4px;font-size:12px;font-weight:600'>MASE &lt; 1.0 — "
        "meilleur que le naïf ✓</span>", unsafe_allow_html=True
    )
    col_leg2.markdown(
        "<span style='background:#ffeb9c;color:#9c5700;padding:3px 8px;"
        "border-radius:4px;font-size:12px;font-weight:600'>1.0–1.3 — acceptable</span>",
        unsafe_allow_html=True
    )
    col_leg3.markdown(
        "<span style='background:#ffc7ce;color:#9c0006;padding:3px 8px;"
        "border-radius:4px;font-size:12px;font-weight:600'>MASE &gt; 1.3 — "
        "pire que le naïf ✗</span>", unsafe_allow_html=True
    )

    # ── Statut ────────────────────────────────────────────────────
    st.subheader("🚦 Statut")

    SMAPE_SEUIL = 30.0
    deploy_ok   = best_smape < SMAPE_SEUIL and best_mase < 1.0

    # Volume anticipé = somme des prédictions du meilleur modèle sur la période de test
    _best_test_preds = bench[best_name]['pred']
    _anticipated_volume = float(np.sum(_best_test_preds))

    d1, d2, d3, d4 = st.columns(4)
    d1.metric(
        "Meilleur modèle", best_name,
        delta=f"SMAPE {best_smape:.1f}%",
        delta_color="normal" if best_smape < SMAPE_SEUIL else "inverse"
    )
    d2.metric(
        "MASE", f"{best_mase:.3f}",
        delta="Meilleur que le naïf ✓" if best_mase < 1 else "Pire que le naïf ✗",
        delta_color="normal" if best_mase < 1 else "inverse"
    )
    d3.metric(
        "Marge de risque prévisionnel",
        f"{best_smape:.1f}%",
        delta="Risque faible ✓" if best_smape < SMAPE_SEUIL else "Risque élevé ✗",
        delta_color="normal" if best_smape < SMAPE_SEUIL else "inverse",
        help="%SMAPE — Erreur relative symétrique. < 30 % = prévision fiable pour la production."
    )
    d4.metric(
        "Volume de réservations anticipé",
        f"{_anticipated_volume:,.0f}",
        help=f"Total des réservations prévues par {best_name} sur la période de test ({len(_best_test_preds)} {label_freq}s)."
    )

    if deploy_ok:
        st.success(
            f"✅ **{best_name} est prêt pour la production** — "
            f"SMAPE {best_smape:.1f}% < {SMAPE_SEUIL}% et MASE {best_mase:.3f} < 1.0."
        )
    # else:
    #     st.error(
    #         f"❌ **Modèles insuffisants** (SMAPE {best_smape:.1f}%). "
    #         "Essayez la granularité mensuelle ou choisissez une catégorie plus active."
    #     )

    # ── Graphique principal — stakeholder-friendly ───────────────
    st.subheader("📈 Prévisions vs Réel")
    st.caption(
        "La zone grise représente l'incertitude (intervalle de confiance 80%) "
        "du meilleur modèle. Plus la zone est étroite, plus le modèle est précis."
    )

    # Couleurs distinctes et propres
    COLOR_HIST  = '#4A9EFF'   # bleu — données historiques
    COLOR_REAL  = '#FFFFFF'   # blanc — réel pendant le test
    COLOR_MAP   = {
        'Naïf':         '#888888',
        'SARIMA':       '#FF6B6B',
        'Holt-Winters': '#51CF66',
        'Prophet':      '#FCC419',
        'LightGBM':     '#CC5DE8',
        'Ensemble':     '#FF922B',
    }

    fig = go.Figure()

    # 1. Données historiques (train)
    fig.add_trace(go.Scatter(
        x=train['ds'], y=train['y'],
        mode='lines', name='Historique',
        line=dict(color=COLOR_HIST, width=2),
        opacity=0.7
    ))

    # 2. Valeurs réelles du test (à prédire)
    fig.add_trace(go.Scatter(
        x=test['ds'], y=test['y'],
        mode='lines+markers', name='Réel (période de test)',
        line=dict(color=COLOR_REAL, width=2.5),
        marker=dict(size=6, symbol='circle')
    ))

    # 3. Séparateur train/test
    fig.add_vline(
        x=train['ds'].iloc[-1].timestamp() * 1000,
        line=dict(color='rgba(200,200,200,0.5)', dash='dot', width=1.5),
        annotation_text="Début du test",
        annotation_font_size=11,
        annotation_position="top left"
    )

    # 4. IC du MEILLEUR modèle uniquement (pas de superposition)
    best_data = bench[best_name]
    if best_data.get('ci') is not None:
        ci_lo, ci_hi = best_data['ci']
        n_test = len(test['ds'])
        ci_len = min(len(ci_lo), len(ci_hi), n_test)
        test_dates = test['ds'].values[:ci_len]
        c_best = COLOR_MAP.get(best_name, '#FCC419')

        fig.add_trace(go.Scatter(
            x=np.concatenate([test_dates, test_dates[::-1]]),
            y=np.concatenate([ci_hi[:ci_len], ci_lo[:ci_len][::-1]]),
            fill='toself',
            fillcolor='rgba(252,196,25,0.15)',
            line=dict(width=0),
            name=f'Incertitude {best_name} (IC 80%)',
            hoverinfo='skip'
        ))

    # 5. Prédictions — chaque modèle
    for name, v in bench.items():
        color = COLOR_MAP.get(name, '#AAAAAA')
        n_p   = len(v['pred'])
        n_t   = len(test['ds'])
        dates = test['ds'].values[:min(n_p, n_t)]
        preds = v['pred'][:min(n_p, n_t)]

        dash  = 'solid' if name == best_name else 'dot'
        width = 2.5 if name == best_name else 1.5
        label = f"{'★ ' if name == best_name else ''}{name} (SMAPE {v['smape']:.1f}%)"

        fig.add_trace(go.Scatter(
            x=dates, y=preds,
            mode='lines',
            name=label,
            line=dict(color=color, dash=dash, width=width),
        ))

    fig.update_layout(
        title=f"Prévision des réservations — {cat}",
        xaxis_title="Date",
        yaxis_title="Réservations",
        template='plotly_dark',
        height=460,
        hovermode='x unified',
        legend=dict(
            orientation='h', y=-0.25, x=0,
            font=dict(size=11)
        ),
        margin=dict(t=50, b=80)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Feature importance LightGBM ─────────────────────────────
    if 'LightGBM' in bench and LGB_OK and 'model' in bench['LightGBM']:
        with st.expander("🔍 Facteurs explicatifs — LightGBM"):
            lgb_data = bench['LightGBM']
            fi = pd.DataFrame({
                'Variable': lgb_data['feat'],
                'Importance': lgb_data['model'].feature_importances_
            }).sort_values('Importance', ascending=True).tail(10)

            fig_fi, ax_fi = plt.subplots(figsize=(8, 4))
            ax_fi.barh(fi['Variable'], fi['Importance'], color='#CC5DE8', height=0.6)
            ax_fi.set_xlabel("Importance (gain)")
            ax_fi.set_title("Top 10 variables les plus influentes")
            ax_fi.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_fi)
            st.caption(
                "Un lag_1 dominant indique une forte autocorrélation : "
                "le volume de la période précédente prédit bien le suivant."
            )

    # ── Validation croisée walk-forward ─────────────────────────
    with st.expander("🔄 Validation croisée (fiabilité en production)"):
        if STATS_OK and n_pts >= 24:
            from sklearn.model_selection import TimeSeriesSplit
            n_folds  = 3
            tscv     = TimeSeriesSplit(n_splits=n_folds)
            cv_rows  = []

            for fold, (tr_i, te_i) in enumerate(tscv.split(df_ts)):
                tr_cv = df_ts.iloc[tr_i]
                te_cv = df_ts.iloc[te_i]
                if len(tr_cv) < 8 or len(te_cv) < 2:
                    continue
                try:
                    fit_cv = SARIMAX(
                        tr_cv['y'], order=(1,1,1),
                        enforce_stationarity=False
                    ).fit(disp=False)
                    p_cv = np.clip(fit_cv.forecast(len(te_cv)).values, 0, None)
                    _, _, sp_cv, ms_cv = metrics(te_cv['y'].values, p_cv, tr_cv['y'].values)
                    cv_rows.append({
                        'Fold': f"#{fold+1}",
                        'Entraînement': f"{len(tr_cv)} {label_freq}s",
                        'Test': f"{len(te_cv)} {label_freq}s",
                        'SMAPE (%)': f"{sp_cv:.1f}%",
                        'MASE': f"{ms_cv:.3f}"
                    })
                except Exception:
                    pass

            if cv_rows:
                cv_df = pd.DataFrame(cv_rows)
                st.dataframe(cv_df, use_container_width=True, hide_index=True)
                smapes_num = [float(r['SMAPE (%)'].rstrip('%')) for r in cv_rows]
                st.write(
                    f"**SMAPE moyen** : {np.mean(smapes_num):.1f}%  "
                    f"± {np.std(smapes_num):.1f}% sur {len(cv_rows)} folds"
                )
                st.caption(
                    "Un écart-type faible entre les folds indique un modèle stable. "
                    "Un écart-type élevé signale une sensibilité aux données."
                )
        else:
            st.info("Nécessite statsmodels et ≥ 24 périodes.")

    # ── Simulateur — prévision future ───────────────────────────
    st.subheader("🔮 Prévision future")
    st.caption("Générer une prévision au-delà des données disponibles.")

    with st.form("sim_future"):
        sf1, sf2 = st.columns(2)
        future_cat = sf1.selectbox(
            "Catégorie à prévoir",
            valid_cats,
            index=valid_cats.index(cat),
            help="Vous pouvez prévoir une autre catégorie que celle affichée dans le benchmark."
        )
        n_fut = sf2.slider(
            "Horizon de prévision",
            4, (24 if USE_MONTHLY else 52),
            (6 if USE_MONTHLY else 12),
            help="Nombre de mois ou semaines à prévoir"
        )
        submit = st.form_submit_button("📊 Calculer la prévision")

    if submit:
        try:
            df_ts_future, zero_pct_future = prepare_series_for_category(future_cat)
            n_pts_future = len(df_ts_future)

            if n_pts_future < MIN_POINTS:
                st.error(
                    f"Pas assez de données pour prévoir {future_cat} "
                    f"({n_pts_future} {label_freq}s, minimum {MIN_POINTS})."
                )
                st.stop()

            if zero_pct_future >= 0.40:
                st.warning(
                    f"{future_cat} contient {zero_pct_future:.0%} de périodes à zéro. "
                    "La prévision reste possible, mais la granularité mensuelle sera plus stable."
                )

            if future_cat == cat:
                future_bench = bench
                future_eligible = eligible
            else:
                _, _, future_bench, future_eligible, _ = (
                    train_benchmark_models(df_ts_future)
                )

            # ── Sélection automatique du meilleur modèle par SMAPE ──
            # On inclut l'Ensemble dans la compétition : le gagnant du benchmark
            # pour cette catégorie est toujours respecté, quel que soit le filtre.
            _all_candidates = [
                n for n in future_bench
                if n != 'Naïf'
                and future_bench[n]['smape'] is not None
                and future_bench[n]['smape'] > 0
            ]
            if _all_candidates:
                mc_name = min(_all_candidates, key=lambda n: future_bench[n]['smape'])
            elif 'Naïf' in future_bench:
                mc_name = 'Naïf'
            else:
                st.error("Aucun modèle disponible pour cette catégorie.")
                st.stop()

            st.info(
                f"🏆 **Modèle auto-sélectionné : {mc_name}** "
                f"(SMAPE {future_bench[mc_name]['smape']:.1f}% — meilleur score pour « {future_cat} »)"
            )

            last_date = df_ts_future['ds'].max()
            fut_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1 if USE_MONTHLY else 0,
                                                weeks=0 if USE_MONTHLY else 1),
                periods=n_fut, freq=freq
            )
            fut_preds = None
            fut_ci = None

            if mc_name == 'SARIMA' and STATS_OK:
                fit_fc = SARIMAX(
                    df_ts_future['y'], order=(1,1,1),
                    enforce_stationarity=False
                ).fit(disp=False)
                fc_obj = fit_fc.get_forecast(n_fut)
                fut_preds = np.clip(fc_obj.predicted_mean.values, 0, None)
                ci_fc = fc_obj.conf_int(alpha=0.20)
                fut_ci = (np.clip(ci_fc.iloc[:, 0].values, 0, None), ci_fc.iloc[:, 1].values)

            elif mc_name == 'Holt-Winters' and STATS_OK:
                period_fut = 12 if USE_MONTHLY else 52
                use_s_fut = len(df_ts_future) >= 2 * period_fut
                hw_fc = ExponentialSmoothing(
                    df_ts_future['y'].values, trend='add',
                    seasonal='add' if use_s_fut else None,
                    seasonal_periods=period_fut if use_s_fut else None,
                    initialization_method='estimated'
                ).fit(optimized=True)
                fut_preds = np.clip(hw_fc.forecast(n_fut), 0, None)
                std_r = np.std(df_ts_future['y'].values - hw_fc.fittedvalues)
                fut_ci = (np.clip(fut_preds - 1.28 * std_r, 0, None), fut_preds + 1.28 * std_r)

            elif mc_name == 'Prophet' and PROPHET_OK and 'cfg' in future_bench.get('Prophet', {}):
                prophet_cfg = future_bench['Prophet']['cfg']
                df_all_p = prepare_prophet_frame(
                    df_ts_future,
                    prophet_cfg['scale'],
                    prophet_cfg['has_mktg'],
                    prophet_cfg['has_vis'],
                    prophet_cfg['mktg_scale'],
                    prophet_cfg['vis_scale']
                )
                prophet_model = build_prophet_model(
                    prophet_cfg['cpp'],
                    prophet_cfg['yearly_s'],
                    prophet_cfg['has_mktg'],
                    prophet_cfg['has_vis']
                )

                with _silence():
                    prophet_model.fit(df_all_p)

                fut_df = pd.DataFrame({'ds': fut_dates})
                if prophet_cfg['has_mktg']:
                    fut_df['mktg_norm'] = df_ts_future['mktg'].mean() / prophet_cfg['mktg_scale']
                if prophet_cfg['has_vis']:
                    fut_df['vis_norm'] = df_ts_future['visitors'].mean() / prophet_cfg['vis_scale']

                fc_p_fut = prophet_model.predict(fut_df)
                fut_preds = np.clip(fc_p_fut['yhat'].values * prophet_cfg['scale'], 0, None)
                fut_ci = (
                    np.clip(fc_p_fut['yhat_lower'].values * prophet_cfg['scale'], 0, None),
                    fc_p_fut['yhat_upper'].values * prophet_cfg['scale']
                )

            elif mc_name == 'LightGBM' and LGB_OK and 'model' in future_bench.get('LightGBM', {}):
                hist_tmp = df_ts_future.copy()
                fut_preds_list = []
                lgb_m_fut = future_bench['LightGBM']['model']
                avail_fut = future_bench['LightGBM']['feat']
                lags_fut = future_bench['LightGBM']['lags']
                for i in range(n_fut):
                    df_lag_tmp = make_lags(hist_tmp, lags_fut)
                    if len(df_lag_tmp) == 0:
                        break
                    last_r = df_lag_tmp.iloc[[-1]][avail_fut]
                    pv = float(np.clip(lgb_m_fut.predict(last_r)[0], 0, None))
                    fut_preds_list.append(pv)
                    new_r = pd.DataFrame([{
                        'ds': fut_dates[i], 'y': pv,
                        'mktg': hist_tmp['mktg'].mean(),
                        'visitors': hist_tmp['visitors'].mean()
                    }])
                    hist_tmp = pd.concat([hist_tmp, new_r], ignore_index=True)
                fut_preds = np.array(fut_preds_list)

            elif mc_name == 'Ensemble':
                ens_preds_tmp = []
                for en, ev in future_eligible.items():
                    if en == 'Prophet' and PROPHET_OK and 'cfg' in ev:
                        prophet_cfg = ev['cfg']
                        df_all_ep = prepare_prophet_frame(
                            df_ts_future,
                            prophet_cfg['scale'],
                            prophet_cfg['has_mktg'],
                            prophet_cfg['has_vis'],
                            prophet_cfg['mktg_scale'],
                            prophet_cfg['vis_scale']
                        )
                        prophet_model = build_prophet_model(
                            prophet_cfg['cpp'],
                            prophet_cfg['yearly_s'],
                            prophet_cfg['has_mktg'],
                            prophet_cfg['has_vis']
                        )
                        with _silence():
                            prophet_model.fit(df_all_ep)
                        fut_ep = pd.DataFrame({'ds': fut_dates})
                        if prophet_cfg['has_mktg']:
                            fut_ep['mktg_norm'] = df_ts_future['mktg'].mean() / prophet_cfg['mktg_scale']
                        if prophet_cfg['has_vis']:
                            fut_ep['vis_norm'] = df_ts_future['visitors'].mean() / prophet_cfg['vis_scale']
                        fc_ep = prophet_model.predict(fut_ep)
                        ens_preds_tmp.append(np.clip(fc_ep['yhat'].values * prophet_cfg['scale'], 0, None))
                    elif en == 'SARIMA' and STATS_OK:
                        fit_e = SARIMAX(
                            df_ts_future['y'], order=(1, 1, 1), enforce_stationarity=False
                        ).fit(disp=False)
                        ens_preds_tmp.append(np.clip(fit_e.forecast(n_fut).values, 0, None))
                    elif en == 'Holt-Winters' and STATS_OK:
                        use_s_e = len(df_ts_future) >= 2 * (12 if USE_MONTHLY else 52)
                        hw_e = ExponentialSmoothing(
                            df_ts_future['y'].values, trend='add',
                            seasonal='add' if use_s_e else None,
                            seasonal_periods=(12 if USE_MONTHLY else 52) if use_s_e else None,
                            initialization_method='estimated'
                        ).fit(optimized=True)
                        ens_preds_tmp.append(np.clip(hw_e.forecast(n_fut), 0, None))
                    elif en == 'LightGBM' and LGB_OK and 'model' in ev:
                        hist_tmp = df_ts_future.copy()
                        lgb_preds_tmp = []
                        for i in range(n_fut):
                            df_lag_tmp = make_lags(hist_tmp, ev['lags'])
                            if len(df_lag_tmp) == 0:
                                break
                            last_r = df_lag_tmp.iloc[[-1]][ev['feat']]
                            pv = float(np.clip(ev['model'].predict(last_r)[0], 0, None))
                            lgb_preds_tmp.append(pv)
                            new_r = pd.DataFrame([{
                                'ds': fut_dates[i], 'y': pv,
                                'mktg': hist_tmp['mktg'].mean(),
                                'visitors': hist_tmp['visitors'].mean()
                            }])
                            hist_tmp = pd.concat([hist_tmp, new_r], ignore_index=True)
                        if lgb_preds_tmp:
                            ens_preds_tmp.append(np.array(lgb_preds_tmp))
                if ens_preds_tmp:
                    min_len = min(len(pred) for pred in ens_preds_tmp)
                    fut_preds = np.mean([pred[:min_len] for pred in ens_preds_tmp], axis=0)

            # ── Graphique prévision future ── propre, stakeholder ──
            if fut_preds is not None and len(fut_preds) > 0:
                n_hist_show = min(24 if USE_MONTHLY else 52, len(df_ts_future))
                hist_recent = df_ts_future.tail(n_hist_show)

                fig_fut = go.Figure()

                # IC uniquement si disponible
                if fut_ci is not None:
                    lo_f, hi_f = fut_ci
                    fig_fut.add_trace(go.Scatter(
                        x=np.concatenate([fut_dates, fut_dates[::-1]]),
                        y=np.concatenate([hi_f, lo_f[::-1]]),
                        fill='toself',
                        fillcolor='rgba(252,196,25,0.15)',
                        line=dict(width=0),
                        name='Incertitude (IC 80%)',
                        hoverinfo='skip'
                    ))
                    # Limites IC comme lignes fines
                    fig_fut.add_trace(go.Scatter(
                        x=fut_dates, y=hi_f,
                        mode='lines', line=dict(color='rgba(252,196,25,0.4)', width=1, dash='dot'),
                        showlegend=False, hoverinfo='skip'
                    ))
                    fig_fut.add_trace(go.Scatter(
                        x=fut_dates, y=lo_f,
                        mode='lines', line=dict(color='rgba(252,196,25,0.4)', width=1, dash='dot'),
                        showlegend=False, hoverinfo='skip'
                    ))

                # Historique récent
                fig_fut.add_trace(go.Scatter(
                    x=hist_recent['ds'], y=hist_recent['y'],
                    mode='lines+markers', name='Historique récent',
                    line=dict(color=COLOR_HIST, width=2),
                    marker=dict(size=5)
                ))

                # Prévision
                fig_fut.add_trace(go.Scatter(
                    x=fut_dates[:len(fut_preds)], y=fut_preds,
                    mode='lines+markers',
                    name=f'Prévision {mc_name}',
                    line=dict(color='#FCC419', width=2.5),
                    marker=dict(size=8, symbol='star')
                ))

                # Ligne "aujourd'hui"
                fig_fut.add_vline(
                    x=last_date.timestamp() * 1000,
                    line=dict(color='rgba(200,200,200,0.5)', dash='dot', width=1.5),
                    annotation_text="Aujourd'hui",
                    annotation_font_size=11,
                    annotation_position="top left"
                )

                fig_fut.update_layout(
                    title=f"Prévision sur {n_fut} {label_freq}s — {future_cat} ({mc_name})",
                    xaxis_title="Date",
                    yaxis_title="Réservations prévues",
                    template='plotly_dark',
                    height=420,
                    hovermode='x unified',
                    legend=dict(orientation='h', y=-0.25, font=dict(size=11)),
                    margin=dict(t=50, b=80)
                )
                st.plotly_chart(fig_fut, use_container_width=True)

                # KPIs de la prévision
                total  = float(np.sum(fut_preds))
                moy    = float(np.mean(fut_preds))
                peak_i = int(np.argmax(fut_preds))
                pk_dt  = fut_dates[peak_i]
                pk_val = float(fut_preds[peak_i])

                kc1, kc2, kc3, kc4 = st.columns(4)
                kc1.metric(f"Total ({n_fut} {label_freq}s)", f"{total:,.0f}")
                kc2.metric(f"Moyenne / {label_freq}", f"{moy:,.0f}")
                kc3.metric("Pic attendu", pk_dt.strftime('%b %Y'))
                kc4.metric("Volume du pic", f"{pk_val:,.0f}")
            else:
                st.error("Impossible de générer la prévision avec le modèle sélectionné.")

        except Exception as ex:
            st.error(f"Erreur simulateur : {ex}")

    # ── Interprétation métier ────────────────────────────────────
    with st.expander("📋 Synthèse et recommandations"):
        best_row = bench_df.loc[best_idx]
        pret = best_row['SMAPE (%)'] < 30 and best_row['MASE'] < 1.0
        st.markdown(f"""
**Résultats — {best_name}**

| Indicateur | Valeur | Seuil | Statut |
|-----------|--------|-------|--------|
| SMAPE     | {best_row['SMAPE (%)']:.1f}% | < 30% pour la production | {"✅" if best_row['SMAPE (%)']<30 else "⚠️"} |
| MASE      | {best_row['MASE']:.3f} | < 1.0 (meilleur que le naïf) | {"✅" if best_row['MASE']<1 else "⚠️"} |

**Recommandations opérationnelles**

1. **Ré-entraînement mensuel** — Les séries temporelles dérivent. Planifier un job automatique chaque début de mois.
2. **Alerte production** — Si le SMAPE sur les 4 dernières périodes dépasse 30 %, déclencher une révision du modèle.
3. **Granularité mensuelle** — Pour les catégories avec beaucoup de zéros hebdomadaires, la granularité mensuelle réduit le bruit et améliore la précision.
4. **Marketing comme signal** — Si les campagnes marketing sont planifiées à l'avance, Prophet et LightGBM peuvent les intégrer comme variable explicative.
5. **Ensemble comme stratégie de fallback** — En cas de doute sur le meilleur modèle, l'Ensemble est souvent plus robuste qu'un seul modèle.

**Statut final** : {"🟢 Prêt pour la production" if pret else "🟡 Améliorations recommandées avant déploiement"}
        """)

# ============================================================
# NLP PAGE
# ============================================================
elif page == "🤖 NLP":
    st.title("🤖 NLP — Analyse de Sentiment & Réclamations")

    engine = try_connect()
    if engine is None:
        st.error("❌ Connexion échouée.")
        st.stop()

    @st.cache_data
    def load_nlp_data(_engine):
        q = """
        SELECT c.subject AS complaint_subject, c.status AS complaint_status,
               f.rating, f.final_price, e.type AS event_type
        FROM "DIM_complaint" c
        JOIN fact_suivi_event f ON f.id_complaint = c.id_complaint
        JOIN dim_event e ON f.event_sk = e.event_sk
        WHERE c.subject IS NOT NULL
        """
        return pd.read_sql(q, _engine)

    try:
        df_nlp = load_nlp_data(engine)
    except Exception as e:
        st.error(f"Erreur chargement NLP : {e}")
        st.stop()

    st.success(f"✅ {len(df_nlp)} réclamations chargées")

    POSITIVE_WORDS = ['excellent','parfait','super','bien','bravo','satisfait',
                      'good','great','perfect','happy','amazing','love','best']
    NEGATIVE_WORDS = ['problème','mauvais','nul','déçu','horrible','pire',
                      'bad','poor','terrible','awful','worst','issue','fail',
                      'complaint','unhappy','wrong','broken','delayed','cancel']

    def score_sentiment_kw(text):
        if not isinstance(text, str): return 0
        t = text.lower()
        pos = sum(1 for w in POSITIVE_WORDS if w in t)
        neg = sum(1 for w in NEGATIVE_WORDS if w in t)
        return 1 if pos > neg else (-1 if neg > pos else 0)

    try:
        from textblob import TextBlob
        df_nlp['polarity'] = df_nlp['complaint_subject'].apply(
            lambda t: TextBlob(t).sentiment.polarity if isinstance(t, str) else 0.0)
        st.caption("✅ TextBlob utilisé")
    except ImportError:
        df_nlp['polarity'] = df_nlp['complaint_subject'].apply(score_sentiment_kw).astype(float)
        st.caption("ℹ️ Analyse par mots-clés (TextBlob non installé)")

    df_nlp['sentiment'] = df_nlp['polarity'].apply(
        lambda p: 'Positif' if p > 0.05 else ('Négatif' if p < -0.05 else 'Neutre'))

    col1, col2, col3 = st.columns(3)
    vc = df_nlp['sentiment'].value_counts()
    col1.metric("😊 Positifs", int(vc.get('Positif', 0)))
    col2.metric("😐 Neutres", int(vc.get('Neutre', 0)))
    col3.metric("😠 Négatifs", int(vc.get('Négatif', 0)))

    st.subheader("Distribution des Sentiments")
    colors_sent = {'Positif': '#27ae60', 'Neutre': '#f39c12', 'Négatif': '#e74c3c'}
    fig_sent, axes_sent = plt.subplots(1, 2, figsize=(12, 4))
    vc_plot = vc.reindex(['Positif', 'Neutre', 'Négatif']).fillna(0)
    axes_sent[0].bar(vc_plot.index, vc_plot.values,
                     color=[colors_sent[k] for k in vc_plot.index])
    axes_sent[0].set_title("Répartition des sentiments")
    axes_sent[0].set_ylabel("Nombre de réclamations")
    if 'event_type' in df_nlp.columns:
        sent_by_type = df_nlp.groupby(['event_type', 'sentiment']).size().unstack(fill_value=0)
        sent_by_type.plot(kind='bar', ax=axes_sent[1],
                          color=[colors_sent.get(c, '#95a5a6') for c in sent_by_type.columns])
        axes_sent[1].set_title("Sentiment par type d'événement")
        axes_sent[1].tick_params(axis='x', rotation=30)
    plt.tight_layout()
    st.pyplot(fig_sent)

    st.subheader("📊 Top 20 mots les plus fréquents")
    from collections import Counter
    import re as _re
    STOP = {'the','a','an','is','in','it','of','and','to','was','for','on','are','with',
            'le','la','les','un','une','de','du','des','et','en','je','il','elle','ne',
            'pas','par','se','this','that','at','be'}
    all_words = []
    for txt in df_nlp['complaint_subject'].dropna():
        all_words.extend([w for w in _re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', txt.lower())
                          if w not in STOP])
    word_freq = Counter(all_words).most_common(20)
    if word_freq:
        wl, cl = zip(*word_freq)
        fig_wf, ax_wf = plt.subplots(figsize=(12, 5))
        ax_wf.barh(list(wl)[::-1], list(cl)[::-1], color='#3498db')
        ax_wf.set_xlabel("Fréquence")
        ax_wf.set_title("Top 20 mots dans les réclamations")
        for i, cnt in enumerate(list(cl)[::-1]):
            ax_wf.text(cnt + 0.2, i, str(cnt), va='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_wf)

    with st.expander("📈 Sentiment vs Note de satisfaction"):
        fig_sr, ax_sr = plt.subplots(figsize=(8, 5))
        for sent, color in colors_sent.items():
            sub = df_nlp[df_nlp['sentiment'] == sent]
            ax_sr.scatter(sub['rating'], sub['polarity'], label=sent, alpha=0.6, color=color, s=40)
        ax_sr.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax_sr.set_xlabel("Note (rating)")
        ax_sr.set_ylabel("Score polarité")
        ax_sr.set_title("Sentiment vs Satisfaction")
        ax_sr.legend()
        plt.tight_layout()
        st.pyplot(fig_sr)

    st.subheader("🔮 Simulateur — Analyser un Texte")
    with st.form("nlp_sim_form"):
        user_text = st.text_area("Texte de réclamation", value="The event was cancelled without notice.", height=100)
        submitted_nlp = st.form_submit_button("🤖 Analyser le sentiment")

    if submitted_nlp and user_text.strip():
        try:
            from textblob import TextBlob as TB
            pol = TB(user_text).sentiment.polarity
            sub_score = TB(user_text).sentiment.subjectivity
            sub_str = f"<p>Subjectivité : <strong>{sub_score:.3f}</strong></p>"
        except ImportError:
            pol = float(score_sentiment_kw(user_text))
            sub_str = ""
        label_nlp = "😊 POSITIF" if pol > 0.05 else ("😠 NÉGATIF" if pol < -0.05 else "😐 NEUTRE")
        color_nlp = "#d4edda" if pol > 0.05 else ("#f8d7da" if pol < -0.05 else "#fff3cd")
        st.markdown(
            f"<div style='background:{color_nlp};padding:16px;border-radius:8px;text-align:center;'>"
            f"<h2>Sentiment : <strong>{label_nlp}</strong></h2>"
            f"<p>Score de polarité : <strong>{pol:.3f}</strong></p>{sub_str}"
            f"</div>", unsafe_allow_html=True)
        fig_pg, ax_pg = plt.subplots(figsize=(8, 1.5))
        c_pol = '#27ae60' if pol > 0 else ('#e74c3c' if pol < 0 else '#f39c12')
        ax_pg.barh(['Polarité'], [max(abs(pol), 0.01)],
                   left=[0.5 - max(abs(pol), 0.01)/2 if pol < 0 else 0.5],
                   color=c_pol, height=0.5)
        ax_pg.set_xlim(0, 1)
        ax_pg.axvline(0.5, color='black', linestyle='--')
        ax_pg.set_title(f"Polarité : {pol:.3f}  (−1=négatif … +1=positif)")
        plt.tight_layout()
        st.pyplot(fig_pg)
        neg_found = [w for w in NEGATIVE_WORDS if w in user_text.lower()]
        if neg_found:
            st.warning(f"🔑 Mots négatifs détectés : **{', '.join(neg_found)}**")

# ============================================================
# RECOMMENDATION PAGE
# ============================================================
elif page == "🔁 Recommandation":
    st.title("🔁 Système de Recommandation d'Événements")

    engine = try_connect()
    if engine is None:
        st.error("❌ Connexion échouée.")
        st.stop()

    @st.cache_data
    def load_reco_data(_engine):
        q = """
        SELECT f.sk_beneficiary, f.event_sk, f.final_price, f.rating,
               f.visitors, f.marketing_spend, f.reservations,
               e.type AS event_type
        FROM fact_suivi_event f
        JOIN dim_event e ON f.event_sk = e.event_sk
        WHERE f.sk_beneficiary IS NOT NULL
        """
        return pd.read_sql(q, _engine)

    try:
        df_reco = load_reco_data(engine)
    except Exception as e:
        st.error(f"Erreur chargement : {e}")
        st.stop()

    from sklearn.metrics.pairwise import cosine_similarity

    st.success(f"✅ {len(df_reco)} interactions chargées")

    le_reco = LabelEncoder()
    df_reco['type_enc'] = le_reco.fit_transform(df_reco['event_type'].fillna('inconnu'))
    event_features = df_reco.groupby('event_sk').agg(
        final_price=('final_price','mean'), rating=('rating','mean'),
        visitors=('visitors','mean'), marketing_spend=('marketing_spend','mean'),
        type_enc=('type_enc','first')).fillna(0)

    sc_reco = StandardScaler()
    ev_matrix = sc_reco.fit_transform(event_features)
    sim_matrix = cosine_similarity(ev_matrix)
    sim_df = pd.DataFrame(sim_matrix, index=event_features.index, columns=event_features.index)
    bene_events = df_reco.groupby('sk_beneficiary')['event_sk'].apply(list).to_dict()
    all_events = event_features.index.tolist()

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Bénéficiaires", len(bene_events))
    col_r2.metric("Événements uniques", len(all_events))
    col_r3.metric("Interactions", len(df_reco))

    with st.expander("🗺️ Similarité inter-événements (top 20)"):
        top_ev = event_features.index[:min(20, len(event_features))]
        fig_sim, ax_sim = plt.subplots(figsize=(10, 8))
        sns.heatmap(sim_df.loc[top_ev, top_ev], cmap='Blues', ax=ax_sim, linewidths=0.3)
        ax_sim.set_title("Similarité cosinus entre événements")
        plt.tight_layout()
        st.pyplot(fig_sim)

    st.subheader("📊 Événements par type")
    type_counts = df_reco.groupby('event_type')['event_sk'].nunique()
    fig_tc, ax_tc = plt.subplots(figsize=(8, 4))
    ax_tc.bar(type_counts.index, type_counts.values, color='#3498db')
    ax_tc.set_xlabel("Type")
    ax_tc.set_ylabel("Événements uniques")
    ax_tc.set_title("Nombre d'événements par type")
    ax_tc.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    st.pyplot(fig_tc)

    st.subheader("🔮 Simulateur — Recommandations")
    bene_list = sorted(bene_events.keys())[:200]
    with st.form("reco_sim_form"):
        rc1, rc2 = st.columns(2)
        selected_bene = rc1.selectbox("Bénéficiaire", bene_list)
        n_reco = rc2.slider("Nombre de recommandations", 3, 10, 5)
        submitted_reco = st.form_submit_button("🔁 Générer les recommandations")

    if submitted_reco:
        try:
            attended = bene_events.get(selected_bene, [])
            not_attended = [e for e in all_events if e not in attended]
            if not attended:
                st.warning("Ce bénéficiaire n'a aucun événement connu.")
            elif not not_attended:
                st.info("Ce bénéficiaire a assisté à tous les événements disponibles.")
            else:
                scores = {}
                for ev in not_attended:
                    if ev in sim_df.columns:
                        sims = [sim_df.loc[ev, a] for a in attended if a in sim_df.index]
                        scores[ev] = float(np.mean(sims)) if sims else 0.0
                top_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_reco]
                st.success(f"✅ Top {n_reco} recommandations pour le bénéficiaire **{selected_bene}**")
                rec_data = []
                for ev_id, score in top_recs:
                    info = event_features.loc[ev_id]
                    ti = int(info['type_enc'])
                    etype = le_reco.classes_[ti] if ti < len(le_reco.classes_) else 'Inconnu'
                    rec_data.append({'Événement ID': ev_id, 'Type': etype,
                                     'Prix moyen (€)': f"{info['final_price']:,.0f}",
                                     'Note moy.': f"{info['rating']:.2f}",
                                     'Score': f"{score:.4f}"})
                st.dataframe(pd.DataFrame(rec_data), use_container_width=True)
                fig_rb, ax_rb = plt.subplots(figsize=(8, 4))
                ids_r = [str(r['Événement ID']) for r in rec_data]
                sc_vals = [float(r['Score']) for r in rec_data]
                ax_rb.barh(ids_r[::-1], sc_vals[::-1], color='#3498db')
                ax_rb.set_xlabel("Score de similarité")
                ax_rb.set_title(f"Recommandations — Bénéficiaire {selected_bene}")
                plt.tight_layout()
                st.pyplot(fig_rb)
        except Exception as ex:
            st.error(f"Erreur recommandation : {ex}")

# ============================================================
# DEEP LEARNING PAGE
# ============================================================
elif page == "🧠 Deep Learning":
    st.title("🧠 Deep Learning — Réseau de Neurones MLP")

    engine = try_connect()
    if engine is None:
        st.error("❌ Connexion échouée.")
        st.stop()

    @st.cache_data
    def load_dl_data(_engine):
        q = """
        SELECT f.sk_beneficiary, f.price, f.budget, f.final_price,
               f.rating, f.visitors, f.marketing_spend, f.id_complaint,
               e.type, e.event_date
        FROM fact_suivi_event f
        JOIN dim_event e ON f.event_sk = e.event_sk
        WHERE f.price > 0 AND f.budget > 0
        """
        df = pd.read_sql(q, _engine)
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
        return df

    try:
        df_dl = load_dl_data(engine)
    except Exception as e:
        st.error(f"Erreur chargement : {e}")
        st.stop()

    from sklearn.neural_network import MLPClassifier, MLPRegressor

    task = st.sidebar.selectbox("Tâche MLP", ["Classification (Fidélité)", "Régression (Prix final)"])
    dl_test_size = st.sidebar.slider("Taille test (%)", 10, 40, 20) / 100
    hl_choice = st.sidebar.selectbox("Architecture", ["(64,32)", "(128,64,32)", "(256,128,64)", "(64,)"])
    hl = tuple(int(x) for x in hl_choice.strip("()").split(",") if x.strip())

    df_dl['month'] = df_dl['event_date'].dt.month.fillna(1).astype(int)
    df_dl['price_budget_ratio'] = df_dl['price'] / (df_dl['budget'] + 0.01)
    df_dl['margin'] = df_dl['final_price'] - df_dl['price']
    df_dl['has_complaint'] = (~df_dl['id_complaint'].isna()).astype(int)
    le_dl = LabelEncoder()
    df_dl['type_enc'] = le_dl.fit_transform(df_dl['type'].fillna('inconnu'))

    FEAT_DL = ['price', 'budget', 'final_price', 'rating', 'visitors',
               'marketing_spend', 'price_budget_ratio', 'margin', 'has_complaint',
               'type_enc', 'month']

    st.success(f"✅ {len(df_dl)} lignes chargées")

    if task == "Classification (Fidélité)":
        def make_loyalty_dl(data):
            lmap = {}
            for b in data['sk_beneficiary'].unique():
                bd = data[data['sk_beneficiary'] == b].sort_values('event_date')
                if len(bd) <= 1: lmap[b] = 0; continue
                fd = bd['event_date'].iloc[0]
                later = bd[(bd['event_date'] > fd) &
                           (bd['event_date'] <= fd + pd.DateOffset(months=6))]
                lmap[b] = 1 if len(later) >= 1 else 0
            data['is_loyal'] = data['sk_beneficiary'].map(lmap)
            return data

        df_dl = make_loyalty_dl(df_dl)
        first_res = df_dl.groupby('sk_beneficiary').first().reset_index()
        X_dl = first_res[FEAT_DL].fillna(0)
        y_dl = first_res['is_loyal'].fillna(0)

        if y_dl.nunique() < 2:
            st.error("⚠️ Une seule classe détectée.")
            st.stop()

        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_dl, y_dl, test_size=dl_test_size, random_state=42, stratify=y_dl)
        sc_dl = StandardScaler()
        X_tr_s = sc_dl.fit_transform(X_tr)
        X_te_s = sc_dl.transform(X_te)

        with st.spinner(f"Entraînement MLP {hl_choice}..."):
            mlp = MLPClassifier(hidden_layer_sizes=hl, max_iter=500, random_state=42,
                                early_stopping=True, validation_fraction=0.1)
            mlp.fit(X_tr_s, y_tr)
            y_pred_dl = mlp.predict(X_te_s)

        from sklearn.metrics import accuracy_score, f1_score
        col_d1, col_d2, col_d3 = st.columns(3)
        col_d1.metric("Accuracy", f"{accuracy_score(y_te, y_pred_dl):.4f}")
        col_d2.metric("F1-Score", f"{f1_score(y_te, y_pred_dl, zero_division=0):.4f}")
        col_d3.metric("Itérations", mlp.n_iter_)

        st.subheader("📉 Courbe de perte")
        fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
        ax_loss.plot(mlp.loss_curve_, color='#e74c3c', linewidth=2, label='Train Loss')
        if hasattr(mlp, 'validation_scores_') and mlp.validation_scores_:
            ax2 = ax_loss.twinx()
            ax2.plot(mlp.validation_scores_, color='#27ae60', linestyle='--',
                     linewidth=2, label='Val Score')
            ax2.set_ylabel("Validation Score")
            ax2.legend(loc='lower right')
        ax_loss.set_xlabel("Itération")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title(f"Apprentissage MLP {hl_choice}")
        ax_loss.legend(loc='upper right')
        ax_loss.grid(True, alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_loss)

        st.subheader("🔮 Simulateur MLP — Prédire la Fidélité")
        with st.form("dl_cls_form"):
            dc1, dc2, dc3 = st.columns(3)
            dc_price  = dc1.number_input("Prix (€)", min_value=0, value=3000, step=100)
            dc_budget = dc2.number_input("Budget (€)", min_value=0, value=15000, step=500)
            dc_final  = dc3.number_input("Prix final (€)", min_value=0, value=5000, step=100)
            dc4, dc5, dc6 = st.columns(3)
            dc_visitors = dc4.number_input("Invités", min_value=0, value=200, step=10)
            dc_rating   = dc5.slider("Note", 1.0, 5.0, 3.5, 0.1)
            dc_mktg     = dc6.number_input("Marketing (€)", min_value=0, value=5000, step=500)
            dc7, dc8 = st.columns(2)
            dc_type  = dc7.selectbox("Type", df_dl['type'].dropna().unique().tolist())
            dc_month = dc8.slider("Mois", 1, 12, 6)
            submitted_dl = st.form_submit_button("🧠 Prédire (MLP)")

        if submitted_dl:
            try:
                te = le_dl.transform([dc_type])[0] if dc_type in le_dl.classes_ else 0
                sim_dl = pd.DataFrame([{
                    'price': dc_price, 'budget': dc_budget, 'final_price': dc_final,
                    'rating': dc_rating, 'visitors': dc_visitors, 'marketing_spend': dc_mktg,
                    'price_budget_ratio': dc_price / (dc_budget + 0.01),
                    'margin': dc_final - dc_price, 'has_complaint': 0,
                    'type_enc': te, 'month': dc_month
                }])
                pred_s = int(mlp.predict(sc_dl.transform(sim_dl[FEAT_DL]))[0])
                prob_s = float(mlp.predict_proba(sc_dl.transform(sim_dl[FEAT_DL]))[0][1])
                lbl = "FIDÈLE" if pred_s == 1 else "NON FIDÈLE"
                col_dl = "#d4edda" if pred_s == 1 else "#f8d7da"
                st.markdown(
                    f"<div style='background:{col_dl};padding:14px;border-radius:8px;text-align:center;'>"
                    f"<h2>🧠 MLP → <strong>{lbl}</strong></h2>"
                    f"<p>Probabilité de fidélité : <strong>{prob_s*100:.1f}%</strong></p>"
                    f"</div>", unsafe_allow_html=True)
            except Exception as ex:
                st.error(f"Erreur : {ex}")

    else:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error
        X_dl = df_dl[FEAT_DL].fillna(0)
        y_dl = df_dl['final_price'].fillna(0)
        X_tr, X_te, y_tr, y_te = train_test_split(X_dl, y_dl, test_size=dl_test_size, random_state=42)
        sc_dl = StandardScaler()
        X_tr_s = sc_dl.fit_transform(X_tr)
        X_te_s = sc_dl.transform(X_te)

        with st.spinner(f"Entraînement MLP Regressor {hl_choice}..."):
            mlp_r = MLPRegressor(hidden_layer_sizes=hl, max_iter=500, random_state=42,
                                 early_stopping=True, validation_fraction=0.1)
            mlp_r.fit(X_tr_s, y_tr)
            y_pred_r = mlp_r.predict(X_te_s)

        col_dr1, col_dr2, col_dr3 = st.columns(3)
        col_dr1.metric("R²", f"{r2_score(y_te, y_pred_r):.4f}")
        col_dr2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_te, y_pred_r)):,.0f}")
        col_dr3.metric("Itérations", mlp_r.n_iter_)

        fig_lossR, ax_lossR = plt.subplots(figsize=(10, 4))
        ax_lossR.plot(mlp_r.loss_curve_, color='#9b59b6', linewidth=2)
        ax_lossR.set_xlabel("Itération")
        ax_lossR.set_ylabel("Loss (MSE)")
        ax_lossR.set_title(f"Apprentissage MLP Regressor {hl_choice}")
        ax_lossR.grid(True, alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_lossR)

        fig_avpR, ax_avpR = plt.subplots(figsize=(8, 5))
        ax_avpR.scatter(y_te, y_pred_r, alpha=0.4, color='#9b59b6')
        mn_r, mx_r = float(y_te.min()), float(y_te.max())
        ax_avpR.plot([mn_r, mx_r], [mn_r, mx_r], 'r--')
        ax_avpR.set_xlabel("Réel")
        ax_avpR.set_ylabel("Prédit")
        ax_avpR.set_title("MLP Regressor — Actual vs Predicted")
        plt.tight_layout()
        st.pyplot(fig_avpR)

# ============================================================
# ANOMALY DETECTION PAGE
# ============================================================
elif page == "🚨 Anomalies":
    st.title("🚨 Détection d'Anomalies — Isolation Forest & LOF")

    engine = try_connect()
    if engine is None:
        st.error("❌ Connexion échouée.")
        st.stop()

    @st.cache_data
    def load_anomaly_data(_engine):
        q = """
        SELECT f.price, f.budget, f.final_price, f.rating,
               f.visitors, f.marketing_spend, e.type AS event_type
        FROM fact_suivi_event f
        JOIN dim_event e ON f.event_sk = e.event_sk
        WHERE f.price > 0 AND f.budget > 0 AND f.final_price > 0
        """
        return pd.read_sql(q, _engine)

    try:
        df_ano = load_anomaly_data(engine)
    except Exception as e:
        st.error(f"Erreur chargement : {e}")
        st.stop()

    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    st.success(f"✅ {len(df_ano)} lignes chargées")

    FEAT_ANO = ['price', 'budget', 'final_price', 'rating', 'visitors', 'marketing_spend']
    X_ano = df_ano[FEAT_ANO].fillna(0)
    contamination = st.sidebar.slider("Taux de contamination (%)", 1, 20, 5) / 100
    algo_ano = st.sidebar.selectbox("Algorithme anomalie", ["Isolation Forest", "LOF"])

    sc_ano = StandardScaler()
    X_ano_s = sc_ano.fit_transform(X_ano)

    with st.spinner("Détection des anomalies..."):
        if algo_ano == "Isolation Forest":
            detector = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
            ano_labels = detector.fit_predict(X_ano_s)
            ano_scores = detector.score_samples(X_ano_s)
        else:
            n_nb = max(5, int(len(X_ano) * 0.05))
            detector = LocalOutlierFactor(contamination=contamination, n_neighbors=n_nb, novelty=True)
            detector.fit(X_ano_s)
            ano_labels = detector.predict(X_ano_s)
            ano_scores = detector.score_samples(X_ano_s)

    df_ano['anomaly'] = ano_labels
    df_ano['ano_score'] = ano_scores
    n_ano = int((ano_labels == -1).sum())
    n_norm = int((ano_labels == 1).sum())

    col_a1, col_a2, col_a3 = st.columns(3)
    col_a1.metric("🔴 Anomalies", n_ano)
    col_a2.metric("🟢 Normaux", n_norm)
    col_a3.metric("Taux", f"{n_ano/len(df_ano)*100:.1f}%")

    from sklearn.decomposition import PCA as _PCA
    pca_ano = _PCA(n_components=2)
    X_2d = pca_ano.fit_transform(X_ano_s)
    df_ano['PC1'] = X_2d[:, 0]
    df_ano['PC2'] = X_2d[:, 1]
    normal_mask = df_ano['anomaly'] == 1

    st.subheader("🗺️ Visualisation PCA 2D")
    fig_ano, ax_ano = plt.subplots(figsize=(10, 6))
    ax_ano.scatter(df_ano.loc[normal_mask, 'PC1'], df_ano.loc[normal_mask, 'PC2'],
                   c='#3498db', s=25, alpha=0.5, label='Normal')
    ax_ano.scatter(df_ano.loc[~normal_mask, 'PC1'], df_ano.loc[~normal_mask, 'PC2'],
                   c='#e74c3c', s=70, alpha=0.8, marker='X', label='Anomalie', zorder=5)
    ax_ano.set_title(f"Anomalies — {algo_ano}")
    ax_ano.set_xlabel("PC1")
    ax_ano.set_ylabel("PC2")
    ax_ano.legend()
    ax_ano.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_ano)

    with st.expander("📊 Distribution des scores"):
        fig_sc, ax_sc = plt.subplots(figsize=(10, 4))
        ax_sc.hist(ano_scores[ano_labels == 1], bins=40, color='#3498db', alpha=0.7, label='Normal')
        ax_sc.hist(ano_scores[ano_labels == -1], bins=20, color='#e74c3c', alpha=0.8, label='Anomalie')
        ax_sc.set_xlabel("Score")
        ax_sc.set_ylabel("Fréquence")
        ax_sc.set_title("Distribution des scores d'anomalie")
        ax_sc.legend()
        plt.tight_layout()
        st.pyplot(fig_sc)

    with st.expander("🔴 Top 20 anomalies"):
        top_ano = df_ano[df_ano['anomaly'] == -1].sort_values('ano_score').head(20)
        st.dataframe(top_ano[FEAT_ANO + ['ano_score']].reset_index(drop=True)
                     .style.format({c: '{:,.0f}' for c in FEAT_ANO if c != 'rating'})
                     .format({'rating': '{:.2f}', 'ano_score': '{:.4f}'}),
                     use_container_width=True)

    st.subheader("🔮 Simulateur — Tester un Profil")
    with st.form("ano_sim_form"):
        as1, as2, as3 = st.columns(3)
        as_price    = as1.number_input("Prix (€)", min_value=0, value=3000, step=100)
        as_budget   = as2.number_input("Budget (€)", min_value=0, value=15000, step=500)
        as_final    = as3.number_input("Prix final (€)", min_value=0, value=5000, step=100)
        as4, as5, as6 = st.columns(3)
        as_rating   = as4.slider("Note", 1.0, 5.0, 3.5, 0.1)
        as_visitors = as5.number_input("Visiteurs", min_value=0, value=300, step=50)
        as_mktg     = as6.number_input("Marketing (€)", min_value=0, value=8000, step=500)
        submitted_ano = st.form_submit_button("🚨 Détecter l'anomalie")

    if submitted_ano:
        try:
            sim_ano_df = pd.DataFrame([{
                'price': as_price, 'budget': as_budget, 'final_price': as_final,
                'rating': as_rating, 'visitors': as_visitors, 'marketing_spend': as_mktg
            }])
            sim_ano_s = sc_ano.transform(sim_ano_df[FEAT_ANO])
            pred_ano = int(detector.predict(sim_ano_s)[0])
            score_ano = float(detector.score_samples(sim_ano_s)[0])

            if pred_ano == -1:
                st.markdown(
                    "<div style='background:#f8d7da;padding:16px;border-radius:8px;text-align:center;'>"
                    "<h2>🔴 ANOMALIE DÉTECTÉE</h2>"
                    "<p>Ce profil présente des caractéristiques inhabituelles.</p>"
                    "</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div style='background:#d4edda;padding:16px;border-radius:8px;text-align:center;'>"
                    "<h2>🟢 PROFIL NORMAL</h2>"
                    "<p>Ce profil est conforme aux données habituelles.</p>"
                    "</div>", unsafe_allow_html=True)

            st.metric("Score d'anomalie", f"{score_ano:.4f}")

            sim_pca_ano = pca_ano.transform(sim_ano_s)
            fig_sim_ano, ax_sim_ano = plt.subplots(figsize=(8, 5))
            ax_sim_ano.scatter(df_ano.loc[normal_mask, 'PC1'], df_ano.loc[normal_mask, 'PC2'],
                               c='#3498db', s=20, alpha=0.4, label='Normal')
            ax_sim_ano.scatter(df_ano.loc[~normal_mask, 'PC1'], df_ano.loc[~normal_mask, 'PC2'],
                               c='#e74c3c', s=50, alpha=0.6, marker='X', label='Anomalie')
            ax_sim_ano.scatter(float(sim_pca_ano[0, 0]), float(sim_pca_ano[0, 1]),
                               c='yellow', s=400, marker='*', edgecolors='black',
                               zorder=10, label='⭐ Votre profil')
            ax_sim_ano.set_title("Position de votre profil dans l'espace PCA")
            ax_sim_ano.legend()
            ax_sim_ano.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig_sim_ano)
        except Exception as ex:
            st.error(f"Erreur détection : {ex}")
