import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface ChatResponse {
  reply: string;
  sql: string | null;
  data: Record<string, unknown>[];
  type: 'kpi' | 'chart' | 'general';
  status: string;
}

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly base = 'http://127.0.0.1:8000';

  predict(data: { attendees: number; duration: number }): Observable<{ prediction: number; status: string }> {
    return this.http.post<{ prediction: number; status: string }>(`${this.base}/predict`, data);
  }

  chatbot(message: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.base}/chatbot`, { message });
  }
}
