// ── FastAPI (port 8000) ──────────────────────────────────────────────────────

export interface PricePredictRequest {
  price: number;
  budget: number;
  marketing_spend: number;
  new_beneficiaries: number;
  reservations: number;
  nb_events: number;
  avg_spent_user: number;
  type?: string;
  status?: string;
}

export interface PricePredictResponse {
  status: string;
  model: string;
  prediction: number;
  input_received: PricePredictRequest;
}

export interface FidelisationRequest {
  price: number;
  budget: number;
  final_price: number;
  rating: number;
  visitors: number;
  marketing_spend: number;
  price_budget_ratio: number;
  margin: number;
  has_complaint: number;   // 0 | 1
  type_encoded: number;    // 0=Corporate Event, 1=Private Party, 2=Wedding, 3=inconnu
  season_encoded: number;  // 0=automne, 1=ete, 2=hiver, 3=printemps
  is_weekend: number;      // 0 | 1
  month: number;           // 1-12
}

export interface FidelisationResponse {
  status: string;
  model: string;
  prediction: number;
  label: string;
  probabilite_fidelite: number;
  niveau_confiance: string;
  action_recommandee: string;
  accuracy?: number;
  recall?: number;
  roc_auc?: number;
}

export interface ForecastRequest {
  category_name: string;
  horizon: number;
}

export interface ForecastResponse {
  status: string;
  category?: string;
  model?: string;
  history?: Array<{
    date: string;
    value: number;
  }>;
  forecast: Array<{
    date: string;
    value: number;
  }>;
  metrics?: {
    mape: number;
    mae: number;
    rmse: number;
    train_points: number;
    test_points: number;
  };
}

export interface SentimentRequest {
  text: string;
}

export interface SentimentResponse {
  status: string;
  text: string;
  polarity: number;
  label: string;
}

export interface RecommendationRequest {
  beneficiary_id: number;
  n_reco?: number;
}

export interface RecommendationResponse {
  status: string;
  beneficiary_id?: number;
  recommendations: number[];
  type: string;
  scores?: number[];
}

export interface AnomalyRecord {
  price: number;
  budget: number;
  final_price: number;
  rating: number;
  visitors: number;
  marketing_spend: number;
  ano_score: number;
}

export interface AnomalyResponse {
  status: string;
  total_count: number;
  anomaly_count: number;
  sample_anomalies: AnomalyRecord[];
  algorithm?: string;
  contamination?: number;
}

export interface DeepLearningResponse {
  status: string;
  model: string;
  prediction: number;
  confidence: number;
  note: string;
  accuracy?: number;
  f1_score?: number;
  auc?: number;
  iterations?: number;
}

// ── Flask (port 5000) ────────────────────────────────────────────────────────

export interface LoyaltyRequest {
  price: number;
  budget: number;
  final_price: number;
  rating: number;
  visitors: number;
  event_date: string;       // ISO date string
  event_type: 'Corporate Event' | 'Private Party' | 'Wedding' | 'inconnu';
  id_complaint?: string | null;
}

export interface LoyaltyResponse {
  is_loyal: number;
  probability: number;
  status: string;
}

export interface ClusterRequest {
  budget: number;
  price: number;
  final_price: number;
  rating: number;
  visitors: number;
  complaint_status: 'open' | 'closed' | '';
  complaint_subject: string;
  event_type: 'Corporate Event' | 'Private Party' | 'Wedding';
  reservation_status: 'confirmed' | 'cancelled' | 'pending';
  algo?: 'kmeans' | 'dbscan';
}

export interface ClusterResponse {
  cluster_id: number;
  cluster_name: string;
  algorithm: string;
  status: string;
  silhouette_score?: number | null;
  davies_bouldin_score?: number | null;
  n_clusters_detected?: number;
  noise_points?: number;
}
