import { Component, OnInit, OnDestroy, inject, signal, AfterViewInit, ElementRef } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { ApiService } from '../services/api.service';
import { AuthService } from '../services/auth.service';
import {
  PricePredictRequest, PricePredictResponse,
  FidelisationRequest, FidelisationResponse,
  LoyaltyRequest, LoyaltyResponse,
  ClusterRequest, ClusterResponse,
  ForecastRequest, ForecastResponse
} from '../models/ml.models';

type Section = 'overview' | 'dashboards' | 'lab' | 'about';
const PROTECTED: Section[] = ['dashboards', 'lab'];

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrl: './home.component.css',
  standalone: false
})
export class HomeComponent implements OnInit, OnDestroy, AfterViewInit {
  private readonly api       = inject(ApiService);
  private readonly sanitizer = inject(DomSanitizer);
  private readonly el        = inject(ElementRef);
  protected readonly auth    = inject(AuthService);

  readonly pbiUrl     = 'https://app.powerbi.com/reportEmbed?reportId=39e494a8-2968-437f-beb2-aaa2fb9b4625&autoAuth=true&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730';
  readonly pbiUrlSafe: SafeResourceUrl = this.sanitizer.bypassSecurityTrustResourceUrl(this.pbiUrl);

  activeSection = signal<Section>('overview');
  selectedModel = signal<string | null>(null);

  // ── Price Prediction ──────────────────────────────────────────────────────
  priceForm: PricePredictRequest = {
    price: 500, budget: 2000, marketing_spend: 300,
    new_beneficiaries: 50, reservations: 80, nb_events: 5,
    avg_spent_user: 120, type: 'Corporate Event', status: 'confirmed'
  };
  priceResult  = signal<PricePredictResponse | null>(null);
  priceLoading = signal(false);
  priceError   = signal<string | null>(null);

  // ── Fidelisation Prediction ───────────────────────────────────────────────
  fidelForm: FidelisationRequest = {
    price: 500, budget: 2000, final_price: 480, rating: 4.2,
    visitors: 120, marketing_spend: 300, price_budget_ratio: 0.25,
    margin: -20, has_complaint: 0, type_encoded: 0,
    season_encoded: 2, is_weekend: 0, month: 6
  };
  fidelResult  = signal<FidelisationResponse | null>(null);
  fidelLoading = signal(false);
  fidelError   = signal<string | null>(null);

  // ── Loyalty Prediction ────────────────────────────────────────────────────
  loyaltyForm: LoyaltyRequest = {
    price: 500, budget: 2000, final_price: 480, rating: 4.2,
    visitors: 120, event_date: new Date().toISOString().split('T')[0],
    event_type: 'Corporate Event', id_complaint: null
  };
  loyaltyResult  = signal<LoyaltyResponse | null>(null);
  loyaltyLoading = signal(false);
  loyaltyError   = signal<string | null>(null);

  // ── Cluster Prediction ────────────────────────────────────────────────────
  clusterForm: ClusterRequest = {
    budget: 2000, price: 500, final_price: 480, rating: 4.2,
    visitors: 120, complaint_status: 'closed', complaint_subject: 'Autre',
    event_type: 'Corporate Event', reservation_status: 'confirmed', algo: 'kmeans'
  };
  clusterResult  = signal<ClusterResponse | null>(null);
  clusterLoading = signal(false);
  clusterError   = signal<string | null>(null);

  // ── Forecasting ──────────────────────────────────────────────────────────
  forecastForm: ForecastRequest = { category_name: '', horizon: 6 };
  forecastResult = signal<ForecastResponse | null>(null);
  forecastLoading = signal(false);
  forecastError = signal<string | null>(null);
  categories = signal<string[]>([]);

  private onHashChange = () => this.readHash();

  ngOnInit(): void {
    this.readHash();
    window.addEventListener('hashchange', this.onHashChange);
    this.loadCategories();
  }

  private loadCategories(): void {
    this.api.getCategories().subscribe({
      next: (cats) => {
        this.categories.set(cats);
        if (cats.length > 0) this.forecastForm.category_name = cats[0];
      }
    });
  }

  ngAfterViewInit(): void {
    this.animateSection(this.activeSection());
  }

  ngOnDestroy(): void {
    window.removeEventListener('hashchange', this.onHashChange);
  }

  private readHash(): void {
    const hash = (window.location.hash.replace('#', '') || 'overview') as Section;
    if (PROTECTED.includes(hash) && !this.auth.isAdmin) {
      this.activeSection.set('overview');
      window.location.hash = 'overview';
    } else {
      this.selectedModel.set(null);
      this.activeSection.set(hash);
      setTimeout(() => this.animateSection(hash), 50);
    }
  }

  private animateSection(section: Section): void {
    if (typeof window === 'undefined') return;
    const gsap = (window as any).gsap;
    if (!gsap) return;
    if (section === 'overview') {
      gsap.from('.hero > *', { opacity: 0, y: 24, stagger: 0.1, duration: 0.55, ease: 'power2.out', clearProps: 'all' });
      gsap.from('.stat-card', { opacity: 0, y: 20, stagger: 0.08, duration: 0.5, ease: 'power2.out', delay: 0.3, clearProps: 'all' });
    } else if (section === 'about') {
      gsap.from('.feature-card', { opacity: 0, y: 24, stagger: 0.08, duration: 0.5, ease: 'power2.out', clearProps: 'all' });
    } else if (section === 'lab') {
      this.animateLabCards();
    }
  }

  private animateLabCards(): void {
    if (typeof window === 'undefined') return;
    const gsap = (window as any).gsap;
    if (!gsap) return;
    gsap.from('.catalog-card', {
      opacity: 0, y: 28, scale: 0.97,
      stagger: 0.07, duration: 0.5,
      ease: 'power2.out', clearProps: 'all'
    });
  }

  selectModel(id: string | null): void {
    if (typeof window === 'undefined') return;
    const gsap = (window as any).gsap;
    this.selectedModel.set(id);
    if (id && gsap) {
      setTimeout(() => {
        gsap.from('.catalog-panel', {
          opacity: 0, y: 16, duration: 0.35, ease: 'power2.out'
        });
        document.querySelector('.catalog-panel')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }, 0);
    }
  }

  isProtectedAndLocked(section: Section): boolean {
    return PROTECTED.includes(section) && !this.auth.isAdmin;
  }

  // ── Actions ───────────────────────────────────────────────────────────────

  runPricePredict(): void {
    this.priceLoading.set(true);
    this.priceResult.set(null);
    this.priceError.set(null);
    this.api.predictPrice(this.priceForm).subscribe({
      next: (res) => { this.priceResult.set(res); this.priceLoading.set(false); this.animateResult('.price-result'); },
      error: (e)  => { this.priceError.set(e?.error?.detail ?? 'API unreachable.'); this.priceLoading.set(false); }
    });
  }

  trainPrice(): void {
    this.priceLoading.set(true);
    this.priceError.set(null);
    this.api.trainPrice().subscribe({
      next: () => {
        this.priceLoading.set(false);
        alert('Price model trained successfully!');
      },
      error: (e) => {
        this.priceError.set(e?.error?.detail?.message ?? 'Training failed.');
        this.priceLoading.set(false);
      }
    });
  }

  runFidelPredict(): void {
    this.fidelLoading.set(true);
    this.fidelResult.set(null);
    this.fidelError.set(null);
    this.api.predictFidelisation(this.fidelForm).subscribe({
      next: (res) => { this.fidelResult.set(res); this.fidelLoading.set(false); this.animateResult('.fidel-result'); },
      error: (e)  => { this.fidelError.set(e?.error?.detail ?? 'API unreachable.'); this.fidelLoading.set(false); }
    });
  }

  trainFidel(): void {
    this.fidelLoading.set(true);
    this.fidelError.set(null);
    this.api.trainFidelisation().subscribe({
      next: () => { 
        this.fidelLoading.set(false); 
        alert('Model trained successfully! You can now run predictions.');
      },
      error: (e) => { 
        this.fidelError.set(e?.error?.detail?.message ?? 'Training failed. Check database connection.'); 
        this.fidelLoading.set(false); 
      }
    });
  }

  runLoyaltyPredict(): void {
    this.loyaltyLoading.set(true);
    this.loyaltyResult.set(null);
    this.loyaltyError.set(null);
    this.api.predictLoyalty(this.loyaltyForm).subscribe({
      next: (res) => { this.loyaltyResult.set(res); this.loyaltyLoading.set(false); this.animateResult('.loyalty-result'); },
      error: (e)  => { this.loyaltyError.set(e?.error?.detail ?? e?.error?.error ?? 'API unreachable.'); this.loyaltyLoading.set(false); }
    });
  }

  runClusterPredict(): void {
    this.clusterLoading.set(true);
    this.clusterResult.set(null);
    this.clusterError.set(null);
    this.api.predictCluster(this.clusterForm).subscribe({
      next: (res) => { this.clusterResult.set(res); this.clusterLoading.set(false); this.animateResult('.cluster-result'); },
      error: (e)  => { this.clusterError.set(e?.error?.detail ?? e?.error?.error ?? 'API unreachable.'); this.clusterLoading.set(false); }
    });
  }

  runForecast(): void {
    this.forecastLoading.set(true);
    this.forecastResult.set(null);
    this.forecastError.set(null);
    this.api.forecast(this.forecastForm).subscribe({
      next: (res) => { 
        this.forecastResult.set(res); 
        this.forecastLoading.set(false); 
        this.animateResult('.forecast-result'); 
      },
      error: (e) => { 
        this.forecastError.set(e?.error?.detail ?? 'Forecast failed.'); 
        this.forecastLoading.set(false); 
      }
    });
  }

  private animateResult(selector: string): void {
    if (typeof window === 'undefined') return;
    const gsap = (window as any).gsap;
    if (!gsap) return;
    setTimeout(() => {
      gsap.from(selector, { opacity: 0, y: 10, duration: 0.4, ease: 'power2.out' });
    }, 0);
  }
}
