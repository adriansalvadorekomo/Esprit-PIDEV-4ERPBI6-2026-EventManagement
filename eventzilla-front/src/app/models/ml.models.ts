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
}

export interface ForecastRequest {
  category_name: string;
  horizon: number;
}

export interface ForecastResponse {
  status: string;
  category: string;
  history: { date: string; value: number }[];
  forecast: { date: string; value: number }[];
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
}
