import {
  Component, signal, inject, ViewChild, ElementRef,
  AfterViewChecked, OnDestroy
} from '@angular/core';
import {
  Chart, BarController, BarElement, CategoryScale, LinearScale,
  LineController, LineElement, PointElement, ArcElement, PieController,
  Tooltip, Legend, Filler
} from 'chart.js';
import { ApiService, ChatResponse } from '../services/api.service';

Chart.register(
  BarController, BarElement, CategoryScale, LinearScale,
  LineController, LineElement, PointElement, ArcElement, PieController,
  Tooltip, Legend, Filler
);

const COLORS = ['#06B6D4','#3B82F6','#6366F1','#8B5CF6','#EC4899','#F59E0B'];

interface Message {
  role: 'user' | 'bot';
  text: string;
  data?: Record<string, unknown>[];
  type?: string;
  chart_type?: string | null;
  loading?: boolean;
  chartId?: string;
}

@Component({
  selector: 'app-chatbot',
  templateUrl: './chatbot.component.html',
  styleUrl: './chatbot.component.css',
  standalone: false
})
export class ChatbotComponent implements AfterViewChecked, OnDestroy {
  @ViewChild('msgEnd') private msgEnd!: ElementRef;
  private readonly api = inject(ApiService);
  private charts = new Map<string, Chart>();

  open     = signal(false);
  input    = signal('');
  messages = signal<Message[]>([
    { role: 'bot', text: "Hi! I'm the EventZella AI Assistant. Ask me anything about your events data." }
  ]);

  readonly quickQuestions = [
    'Show events by season',
    'Top 5 providers by reservations',
    'Show reservations by status',
    'What is the average rating?',
    'Which event category has the most events?',
    'Show visitor trend by month',
    'What is the total revenue?',
    'Which providers have the lowest ratings?',
  ];

  toggle(): void { this.open.update(v => !v); }
  tableKeys(row: Record<string, unknown>): string[] { return Object.keys(row); }
  ask(q: string): void { this.input.set(q); this.send(); }

  send(): void {
    const text = this.input().trim();
    if (!text) return;
    this.messages.update(m => [...m, { role: 'user', text }]);
    this.input.set('');
    this.messages.update(m => [...m, { role: 'bot', text: '', loading: true }]);

    this.api.chatbot(text).subscribe({
      next: (res: ChatResponse) => {
        const chartId = res.type === 'chart' && res.data?.length
          ? `chart-${Date.now()}` : undefined;

        const msg: Message = {
          role: 'bot', text: res.reply,
          data: res.data?.length ? res.data.slice(0, 20) : undefined,
          type: res.type, chart_type: res.chart_type,
          loading: false, chartId,
        };

        this.messages.update(msgs => {
          const u = [...msgs]; u[u.length - 1] = msg; return u;
        });

        if (chartId) setTimeout(() => this.tryRender(chartId, msg), 80);
      },
      error: () => {
        this.messages.update(msgs => {
          const u = [...msgs];
          u[u.length - 1] = { role: 'bot', text: 'Sorry, could not process your request.', loading: false };
          return u;
        });
      }
    });
  }

  private tryRender(chartId: string, msg: Message, attempt = 0): void {
    if (this.charts.has(chartId)) return;
    const canvas = document.getElementById(chartId) as HTMLCanvasElement | null;
    if (canvas) {
      this.renderChart(canvas, chartId, msg);
    } else if (attempt < 20) {
      setTimeout(() => this.tryRender(chartId, msg, attempt + 1), 50);
    }
  }

  private renderChart(canvas: HTMLCanvasElement, chartId: string, msg: Message): void {
    const data  = msg.data!;
    const keys  = Object.keys(data[0]);
    const labels = data.map(r => String(r[keys[0]]));
    const values = data.map(r => Number(r[keys[1]]));
    const isPie  = msg.chart_type === 'pie';
    const isLine = msg.chart_type === 'line';
    const isHBar = msg.chart_type === 'horizontalBar';

    const baseDataset: any = {
      data: values,
      backgroundColor: isPie
        ? COLORS.slice(0, values.length)
        : isLine ? 'rgba(6,182,212,0.15)' : COLORS.map(c => c + 'cc'),
      borderColor: isPie ? '#0B0E1A' : isLine ? '#06B6D4' : COLORS,
      borderWidth: isPie ? 2 : 2,
      borderRadius: isPie ? 0 : 4,
      fill: isLine ? true : undefined,
      tension: isLine ? 0.4 : undefined,
      pointBackgroundColor: isLine ? '#06B6D4' : undefined,
      pointRadius: isLine ? 4 : undefined,
    };

    const chart = new Chart(canvas, {
      type: isPie ? 'pie' : isLine ? 'line' : 'bar',
      data: { labels, datasets: [baseDataset] },
      options: {
        indexAxis: isHBar ? 'y' : 'x',
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 500 },
        plugins: {
          legend: { display: isPie, labels: { color: '#c9d1d9', font: { size: 11 }, padding: 12 } },
          tooltip: {
            backgroundColor: 'rgba(13,17,23,0.95)',
            titleColor: '#e6edf3',
            bodyColor: '#8b949e',
            borderColor: 'rgba(255,255,255,0.1)',
            borderWidth: 1,
          },
        },
        scales: isPie ? {} : {
          x: {
            ticks: { color: '#8b949e', font: { size: 11 }, maxRotation: 35 },
            grid: { color: 'rgba(255,255,255,0.05)' },
          },
          y: {
            ticks: { color: '#8b949e', font: { size: 11 } },
            grid: { color: 'rgba(255,255,255,0.05)' },
          },
        },
      },
    });

    this.charts.set(chartId, chart);
  }

  onKeydown(e: KeyboardEvent): void { if (e.key === 'Enter') this.send(); }
  ngAfterViewChecked(): void { this.msgEnd?.nativeElement?.scrollIntoView({ behavior: 'smooth' }); }

  ngOnDestroy(): void {
    this.charts.forEach(c => c.destroy());
    this.charts.clear();
  }
}
