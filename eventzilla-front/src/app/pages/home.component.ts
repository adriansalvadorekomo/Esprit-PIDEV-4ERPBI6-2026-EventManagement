import { Component, OnInit, OnDestroy, inject, signal, AfterViewInit, ElementRef } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { ApiService } from '../services/api.service';
import { AuthService } from '../services/auth.service';
import {
  PricePredictRequest, PricePredictResponse,
  FidelisationRequest, FidelisationResponse,
  LoyaltyRequest, LoyaltyResponse,
  ClusterRequest, ClusterResponse,
  ForecastRequest, ForecastResponse,
  SentimentResponse,
  RecommendationResponse,
  AnomalyResponse, DeepLearningResponse
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

  readonly pbiUrl     = 'https://app.powerbi.com/reportEmbed?reportId=de8ac328-20ee-4ad9-8c5a-6cd4d7708ab6&autoAuth=true&embeddedDemo=true';
  readonly pbiUrlSafe: SafeResourceUrl = this.sanitizer.bypassSecurityTrustResourceUrl(this.pbiUrl);

  activeSection = signal<Section>('overview');
  selectedModel = signal<string | null>(null);
  modelInfoVisible = signal(false);
  modelInfoKey = signal<string | null>(null);

  readonly modelInfo: Record<string, { title: string; goal: string; fields: { name: string; desc: string }[]; benchmarks?: { name: string; desc: string; example: string }[] }> = {
    price: {
      title: 'Price Prediction',
      goal: 'This tool estimates the best final price for an event based on its characteristics. Think of it as a smart pricing advisor — you give it the event details and it tells you what price to expect.',
      fields: [
        { name: 'Price', desc: 'The base price you plan to charge.' },
        { name: 'Budget', desc: 'Total budget allocated for the event.' },
        { name: 'Marketing Spend', desc: 'How much you spent promoting the event.' },
        { name: 'New Beneficiaries', desc: 'Number of new attendees expected.' },
        { name: 'Reservations', desc: 'Number of bookings already made.' },
        { name: 'Nb Events', desc: 'How many events the organizer has run before.' },
        { name: 'Avg Spent / User', desc: 'On average, how much each attendee spends.' },
        { name: 'Type', desc: 'The kind of event (Corporate, Wedding, Party…).' },
        { name: 'Status', desc: 'Whether the event is confirmed, pending, or cancelled.' },
      ]
    },
    fidel: {
      title: 'Fidelisation',
      goal: 'This tool predicts whether a client will come back and book again. It gives you a loyalty score and tells you what action to take — like sending a personalized offer or just a newsletter.',
      fields: [
        { name: 'Price / Budget / Final Price', desc: 'The financial profile of the event.' },
        { name: 'Rating', desc: 'How the client rated the event (0 to 5).' },
        { name: 'Visitors', desc: 'How many people attended.' },
        { name: 'Marketing Spend', desc: 'Budget spent on promotion.' },
        { name: 'Has Complaint', desc: 'Did the client file a complaint? Yes or No.' },
        { name: 'Event Type', desc: 'Corporate, Wedding, Private Party, or Unknown.' },
        { name: 'Season', desc: 'Which season the event took place in.' },
        { name: 'Is Weekend', desc: 'Was the event on a weekend?' },
        { name: 'Month', desc: 'Which month the event happened.' },
      ],
      benchmarks: [
        { name: 'Accuracy', desc: 'Out of all clients, how many did the model correctly classify as loyal or not loyal.', example: '85% accuracy means 85 out of 100 predictions were correct.' },
        { name: 'Recall', desc: 'Out of all truly loyal clients, how many did the model successfully identify. High recall means fewer loyal clients are missed.', example: '78% recall means the model caught 78 out of every 100 genuinely loyal clients.' },
        { name: 'ROC-AUC', desc: 'A score from 0 to 1 measuring how well the model separates loyal from non-loyal clients. 0.5 is random guessing; 1.0 is perfect.', example: '0.91 AUC means the model is excellent at ranking loyal clients above non-loyal ones.' },
      ]
    },
    loyalty: {
      title: 'Loyalty Score',
      goal: 'Similar to Fidelisation but uses a different calculation method. It gives a simple Yes/No answer: will this client be loyal? Plus a confidence percentage.',
      fields: [
        { name: 'Price / Budget / Final Price', desc: 'The financial profile of the event.' },
        { name: 'Rating', desc: 'Client satisfaction score (0 to 5).' },
        { name: 'Visitors', desc: 'Number of attendees.' },
        { name: 'Event Date', desc: 'When the event took place.' },
        { name: 'Event Type', desc: 'The category of the event.' },
      ]
    },
    cluster: {
      title: 'Clustering',
      goal: 'This tool groups events into categories based on their profile. It helps you understand what "type" of event you are dealing with — Premium, Potential, or At-Risk — so you can treat each group differently.',
      fields: [
        { name: 'Budget / Price / Final Price', desc: 'The financial profile of the event.' },
        { name: 'Rating', desc: 'Client satisfaction score.' },
        { name: 'Visitors', desc: 'Number of attendees.' },
        { name: 'Algorithm', desc: 'KMeans groups into fixed clusters; DBSCAN finds natural groups automatically.' },
        { name: 'Event Type', desc: 'The category of the event.' },
        { name: 'Reservation Status', desc: 'Whether the booking is confirmed, pending, or cancelled.' },
      ],
      benchmarks: [
        { name: 'Silhouette Score', desc: 'Measures how well each event fits its assigned group vs. other groups. Ranges from -1 to 1. Higher is better.', example: '0.62 means events in the same cluster are clearly more similar to each other than to events in other clusters.' },
        { name: 'Davies-Bouldin Score', desc: 'Measures how spread out and separated the clusters are. Lower is better — it means clusters are compact and far apart.', example: '0.85 is a good score; 2.5 would mean the clusters overlap too much.' },
        { name: 'Clusters Detected', desc: 'The number of distinct groups the algorithm found in your data.', example: '3 clusters might represent Premium, Standard, and At-Risk event profiles.' },
      ]
    },
    forecast: {
      title: 'Demand Forecasting',
      goal: 'This tool looks at past booking history and predicts how many reservations to expect in the coming months. Like a weather forecast, but for your event demand.',
      fields: [
        { name: 'Category', desc: 'The type of events to forecast (e.g. Concerts, Weddings…).' },
        { name: 'Horizon', desc: 'How many months ahead you want to predict.' },
      ],
      benchmarks: [
        { name: 'MAPE', desc: 'Mean Absolute Percentage Error — the average % gap between predicted and actual reservations. Lower is better.', example: '12% MAPE means the forecast is off by about 12 reservations for every 100 expected.' },
        { name: 'MAE', desc: 'Mean Absolute Error — the average number of reservations the forecast misses per month, regardless of direction.', example: 'MAE of 8 means the model is typically off by 8 reservations per month.' },
        { name: 'RMSE', desc: 'Root Mean Square Error — similar to MAE but penalizes large errors more heavily. Useful for spotting months with big misses.', example: 'RMSE of 15 with MAE of 8 means there are occasional months with much larger errors.' },
      ]
    },
    sentiment: {
      title: 'Sentiment Analysis',
      goal: 'Paste any client review or feedback text and this tool instantly tells you if it is Positive, Negative, or Neutral. No reading required — it reads it for you.',
      fields: [
        { name: 'Review Text', desc: 'Any written feedback from a client, in any language.' },
      ]
    },
    reco: {
      title: 'Event Recommendations',
      goal: 'Given a client ID, this tool suggests events they are most likely to enjoy based on what similar clients have attended. Like a "You might also like…" feature.',
      fields: [
        { name: 'Beneficiary ID', desc: 'The unique ID of the client in your system.' },
        { name: 'Count', desc: 'How many event suggestions you want (1 to 10).' },
      ]
    },
    anomaly: {
      title: 'Anomaly Detection',
      goal: 'This tool scans all your event financial data and flags anything that looks unusual — events with abnormal prices, budgets, or visitor numbers. Think of it as a fraud or error detector.',
      fields: [
        { name: '(No inputs needed)', desc: 'The tool automatically analyzes all events in the database.' },
      ]
    },
    dl: {
      title: 'Deep Learning (MLP)',
      goal: 'A more powerful version of the Fidelisation model. It uses a neural network — a system inspired by the human brain — to predict loyalty. It takes the same inputs as Fidelisation but can detect more complex patterns.',
      fields: [
        { name: '(Same inputs as Fidelisation)', desc: 'Fill the Fidelisation form first, then run this model for a second opinion.' },
      ],
      benchmarks: [
        { name: 'Accuracy', desc: 'The percentage of clients correctly classified as loyal or not loyal during testing.', example: '88% accuracy means 88 out of 100 test predictions were correct.' },
        { name: 'F1 Score', desc: 'A balance between catching loyal clients (recall) and not falsely labeling non-loyal ones (precision). Useful when the data is imbalanced.', example: '0.84 F1 means the model is both precise and thorough in identifying loyal clients.' },
        { name: 'AUC', desc: 'Area Under the Curve — how well the model ranks loyal clients above non-loyal ones. 1.0 is perfect; 0.5 is random.', example: '0.93 AUC means the model almost always ranks a truly loyal client higher than a non-loyal one.' },
        { name: 'Iterations', desc: 'How many training rounds the neural network completed before converging. More is not always better — early stopping prevents overfitting.', example: '47 iterations means the network found its optimal weights after 47 passes through the training data.' },
      ]
    },
  };

  openModelInfo(key: string, event: MouseEvent): void {
    event.stopPropagation();
    this.modelInfoKey.set(key);
    this.modelInfoVisible.set(true);
  }

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

  // ── NEW: Sentiment ────────────────────────────────────────────────────────
  sentimentText = 'The event was absolutely fantastic!';
  sentimentResult = signal<SentimentResponse | null>(null);
  sentimentLoading = signal(false);

  // ── NEW: Recommendation ───────────────────────────────────────────────────
  recoBeneId = 1;
  recoCount = 5;
  recoResult = signal<RecommendationResponse | null>(null);
  recoLoading = signal(false);
  recoError = signal<string | null>(null);

  // ── NEW: Anomalies ────────────────────────────────────────────────────────
  anomalyResult = signal<AnomalyResponse | null>(null);
  anomalyLoading = signal(false);
  anomalyError = signal<string | null>(null);

  // ── NEW: Deep Learning (MLP) ──────────────────────────────────────────────
  dlResult = signal<DeepLearningResponse | null>(null);
  dlLoading = signal(false);
  dlError = signal<string | null>(null);

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

  navigate(section: string): void {
    window.location.hash = section;
  }

  isProtectedAndLocked(section: Section): boolean {
    return PROTECTED.includes(section) && !this.auth.isAdmin;
  }

  get forecastTotalReservations(): number {
    return this.forecastResult()?.forecast.reduce((sum, p) => sum + p.value, 0) ?? 0;
  }

  get forecastSeries(): Array<{ label: string; value: number; phase: 'history' | 'forecast' }> {
    const result = this.forecastResult();
    if (!result) return [];
    const history = (result.history ?? []).map((item) => ({
      label: item.date,
      value: item.value,
      phase: 'history' as const
    }));
    const forecast = result.forecast.map((item) => ({
      label: item.date,
      value: item.value,
      phase: 'forecast' as const
    }));
    return [...history, ...forecast];
  }

  get forecastMaxValue(): number {
    const values = this.forecastSeries.map((item) => item.value);
    return values.length ? Math.max(...values, 1) : 1;
  }

  get forecastPolylinePoints(): string {
    const series = this.forecastSeries;
    if (!series.length) return '';
    return series
      .map((item, index) => {
        const x = series.length === 1 ? 0 : (index / (series.length - 1)) * 100;
        const y = 100 - (item.value / this.forecastMaxValue) * 100;
        return `${x},${y}`;
      })
      .join(' ');
  }

  get forecastDividerX(): number | null {
    const historyLength = this.forecastResult()?.history?.length ?? 0;
    const total = this.forecastSeries.length;
    if (!historyLength || historyLength >= total) return null;
    return ((historyLength - 1) / (total - 1)) * 100;
  }

  get anomalyRatePercent(): number | null {
    const result = this.anomalyResult();
    if (!result || !result.total_count) return null;
    return (result.anomaly_count / result.total_count) * 100;
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

  runSentiment(): void {
    this.sentimentLoading.set(true);
    this.sentimentResult.set(null);
    this.api.predictSentiment(this.sentimentText).subscribe({
      next: (res) => {
        this.sentimentResult.set(res);
        this.sentimentLoading.set(false);
        this.animateResult('.sentiment-result');
      },
      error: () => this.sentimentLoading.set(false)
    });
  }

  runReco(): void {
    this.recoLoading.set(true);
    this.recoResult.set(null);
    this.recoError.set(null);
    this.api.recommendEvents(this.recoBeneId, this.recoCount).subscribe({
      next: (res) => {
        this.recoResult.set(res);
        this.recoLoading.set(false);
        this.animateResult('.reco-result');
      },
      error: (e) => {
        this.recoError.set(e?.error?.detail ?? 'Recommendation request failed.');
        this.recoLoading.set(false);
      }
    });
  }

  runAnomalies(): void {
    this.anomalyLoading.set(true);
    this.anomalyResult.set(null);
    this.anomalyError.set(null);
    this.api.detectAnomalies().subscribe({
      next: (res) => {
        this.anomalyResult.set(res);
        this.anomalyLoading.set(false);
        this.animateResult('.anomaly-result');
      },
      error: (e) => {
        this.anomalyError.set(e?.error?.detail ?? 'Anomaly detection failed.');
        this.anomalyLoading.set(false);
      }
    });
  }

  runDL(): void {
    this.dlLoading.set(true);
    this.dlResult.set(null);
    this.dlError.set(null);
    this.api.predictDeepLearning(this.fidelForm).subscribe({
      next: (res) => {
        this.dlResult.set(res);
        this.dlLoading.set(false);
        this.animateResult('.dl-result');
      },
      error: (e) => {
        this.dlError.set(e?.error?.detail ?? 'Deep learning prediction failed.');
        this.dlLoading.set(false);
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
