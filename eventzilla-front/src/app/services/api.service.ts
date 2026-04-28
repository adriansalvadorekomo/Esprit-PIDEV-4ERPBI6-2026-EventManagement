import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  PricePredictRequest, PricePredictResponse,
  FidelisationRequest, FidelisationResponse,
  LoyaltyRequest, LoyaltyResponse,
  ClusterRequest, ClusterResponse,
  ForecastRequest, ForecastResponse,
  SentimentResponse,
  RecommendationResponse,
  AnomalyResponse,
  DeepLearningResponse
} from '../models/ml.models';

export interface ChatResponse {
  reply: string;
  sql: string | null;
  data: Record<string, unknown>[];
  type: 'kpi' | 'chart' | 'general';
  status: string;
}

interface AppConfig { fastapiUrl: string; flaskUrl: string; }

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);
  private fastapi = 'http://localhost:8000';
  private flask   = 'http://localhost:5000';

  constructor() {
    this.http.get<AppConfig>('/config.json').subscribe(cfg => {
      this.fastapi = cfg.fastapiUrl;
      this.flask   = cfg.flaskUrl;
    });
  }

  // ── FastAPI ──────────────────────────────────────────────────────────────

  predictPrice(data: PricePredictRequest): Observable<PricePredictResponse> {
    return this.http.post<PricePredictResponse>(`${this.fastapi}/predict/price`, data);
  }

  predictFidelisation(data: FidelisationRequest): Observable<FidelisationResponse> {
    return this.http.post<FidelisationResponse>(`${this.fastapi}/predict/fidelisation`, data);
  }

  trainPrice(): Observable<{ status: string; model: string; output: string }> {
    return this.http.post<{ status: string; model: string; output: string }>(`${this.fastapi}/train/price`, {});
  }

  trainFidelisation(): Observable<unknown> {
    return this.http.post(`${this.fastapi}/train/fidelisation`, {});
  }

  chatbot(message: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.fastapi}/chatbot`, { message });
  }

  getCategories(): Observable<string[]> {
    return this.http.get<string[]>(`${this.fastapi}/categories`);
  }

  forecast(data: ForecastRequest): Observable<ForecastResponse> {
    return this.http.post<ForecastResponse>(`${this.fastapi}/predict/forecast`, data);
  }

  predictSentiment(text: string): Observable<SentimentResponse> {
    return this.http.post<SentimentResponse>(`${this.fastapi}/predict/sentiment`, { text });
  }

  recommendEvents(beneficiaryId: number, nReco: number = 5): Observable<RecommendationResponse> {
    return this.http.post<RecommendationResponse>(`${this.fastapi}/recommend/events?beneficiary_id=${beneficiaryId}&n_reco=${nReco}`, {});
  }

  detectAnomalies(): Observable<AnomalyResponse> {
    return this.http.post<AnomalyResponse>(`${this.fastapi}/predict/anomalies`, {});
  }

  predictDeepLearning(data: FidelisationRequest): Observable<DeepLearningResponse> {
    return this.http.post<DeepLearningResponse>(`${this.fastapi}/predict/deep-learning`, data);
  }

  // ── Flask ────────────────────────────────────────────────────────────────

  predictLoyalty(data: LoyaltyRequest): Observable<LoyaltyResponse> {
    return this.http.post<LoyaltyResponse>(`${this.flask}/predict-loyalty`, data);
  }

  predictCluster(data: ClusterRequest): Observable<ClusterResponse> {
    return this.http.post<ClusterResponse>(`${this.flask}/predict-cluster`, data);
  }
}
