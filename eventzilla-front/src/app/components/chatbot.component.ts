import {
  Component, signal, inject, ViewChild, ElementRef,
  AfterViewChecked, OnDestroy
} from '@angular/core';
import { ApiService, ChatResponse } from '../services/api.service';

const COLORS = ['#06B6D4', '#3B82F6', '#6366F1', '#8B5CF6', '#EC4899', '#F59E0B'];
const BG = '#0B0E1A';
const GRID = 'rgba(255,255,255,0.06)';
const TEXT = '#8b949e';

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
  private rendered = new Set<string>();

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

        if (chartId) setTimeout(() => this.tryRender(chartId, msg), 60);
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
    if (this.rendered.has(chartId)) return;
    const canvas = document.getElementById(chartId) as HTMLCanvasElement | null;
    if (canvas && canvas.clientWidth > 0) {
      this.rendered.add(chartId);
      this.drawChart(canvas, msg);
    } else if (attempt < 20) {
      setTimeout(() => this.tryRender(chartId, msg, attempt + 1), 30);
    }
  }

  private drawChart(canvas: HTMLCanvasElement, msg: Message): void {
    const data = msg.data!;
    const keys = Object.keys(data[0]);
    const labels = data.map(r => String(r[keys[0]]));
    const values = data.map(r => Number(r[keys[1]]));

    // Set canvas resolution to match display size
    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    canvas.width  = W * devicePixelRatio;
    canvas.height = H * devicePixelRatio;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(devicePixelRatio, devicePixelRatio);

    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, W, H);

    switch (msg.chart_type) {
      case 'pie':         this.drawPie(ctx, W, H, labels, values); break;
      case 'line':        this.drawLine(ctx, W, H, labels, values); break;
      case 'horizontalBar': this.drawHBar(ctx, W, H, labels, values); break;
      default:            this.drawBar(ctx, W, H, labels, values); break;
    }
  }

  // ── Bar ───────────────────────────────────────────────────────────────
  private drawBar(ctx: CanvasRenderingContext2D, W: number, H: number, labels: string[], values: number[]): void {
    const pad = { top: 20, right: 16, bottom: 36, left: 48 };
    const cW = W - pad.left - pad.right;
    const cH = H - pad.top - pad.bottom;
    const max = Math.max(...values) || 1;
    const n = values.length;
    const barW = (cW / n) * 0.6;
    const gap   = cW / n;

    // Grid lines
    ctx.strokeStyle = GRID; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + cH - (i / 4) * cH;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cW, y); ctx.stroke();
      ctx.fillStyle = TEXT; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
      ctx.fillText(this.fmt(max * i / 4), pad.left - 6, y + 4);
    }

    // Bars
    values.forEach((v, i) => {
      const bH = (v / max) * cH;
      const x  = pad.left + i * gap + (gap - barW) / 2;
      const y  = pad.top + cH - bH;
      const color = COLORS[i % COLORS.length];

      // Gradient fill
      const grad = ctx.createLinearGradient(0, y, 0, y + bH);
      grad.addColorStop(0, color);
      grad.addColorStop(1, color + '55');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.roundRect(x, y, barW, bH, 3);
      ctx.fill();

      // Label
      ctx.fillStyle = TEXT; ctx.font = '9px sans-serif'; ctx.textAlign = 'center';
      const lbl = labels[i].length > 9 ? labels[i].slice(0, 8) + '…' : labels[i];
      ctx.fillText(lbl, x + barW / 2, pad.top + cH + 14);
    });
  }

  // ── Horizontal Bar ────────────────────────────────────────────────────
  private drawHBar(ctx: CanvasRenderingContext2D, W: number, H: number, labels: string[], values: number[]): void {
    const pad = { top: 10, right: 20, bottom: 10, left: 90 };
    const cW = W - pad.left - pad.right;
    const cH = H - pad.top - pad.bottom;
    const max = Math.max(...values) || 1;
    const n = values.length;
    const barH = (cH / n) * 0.55;
    const gap  = cH / n;

    values.forEach((v, i) => {
      const bW = (v / max) * cW;
      const y  = pad.top + i * gap + (gap - barH) / 2;
      const color = COLORS[i % COLORS.length];

      const grad = ctx.createLinearGradient(pad.left, 0, pad.left + bW, 0);
      grad.addColorStop(0, color);
      grad.addColorStop(1, color + '55');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.roundRect(pad.left, y, bW, barH, 3);
      ctx.fill();

      // Label left
      ctx.fillStyle = TEXT; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
      const lbl = labels[i].length > 12 ? labels[i].slice(0, 11) + '…' : labels[i];
      ctx.fillText(lbl, pad.left - 6, y + barH / 2 + 4);

      // Value right
      ctx.fillStyle = '#c9d1d9'; ctx.textAlign = 'left';
      ctx.fillText(this.fmt(v), pad.left + bW + 4, y + barH / 2 + 4);
    });
  }

  // ── Line ──────────────────────────────────────────────────────────────
  private drawLine(ctx: CanvasRenderingContext2D, W: number, H: number, labels: string[], values: number[]): void {
    const pad = { top: 20, right: 16, bottom: 36, left: 48 };
    const cW = W - pad.left - pad.right;
    const cH = H - pad.top - pad.bottom;
    const max = Math.max(...values) || 1;
    const min = Math.min(...values);
    const range = max - min || 1;
    const n = values.length;

    const px = (i: number) => pad.left + (i / (n - 1)) * cW;
    const py = (v: number) => pad.top + cH - ((v - min) / range) * cH;

    // Grid
    ctx.strokeStyle = GRID; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (i / 4) * cH;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cW, y); ctx.stroke();
      ctx.fillStyle = TEXT; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
      ctx.fillText(this.fmt(max - (range * i / 4)), pad.left - 6, y + 4);
    }

    // Area fill
    const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + cH);
    grad.addColorStop(0, 'rgba(6,182,212,0.25)');
    grad.addColorStop(1, 'rgba(6,182,212,0.02)');
    ctx.beginPath();
    ctx.moveTo(px(0), py(values[0]));
    values.forEach((v, i) => { if (i > 0) ctx.lineTo(px(i), py(v)); });
    ctx.lineTo(px(n - 1), pad.top + cH);
    ctx.lineTo(px(0), pad.top + cH);
    ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();

    // Line
    ctx.beginPath(); ctx.strokeStyle = '#06B6D4'; ctx.lineWidth = 2;
    values.forEach((v, i) => i === 0 ? ctx.moveTo(px(i), py(v)) : ctx.lineTo(px(i), py(v)));
    ctx.stroke();

    // Dots + labels
    values.forEach((v, i) => {
      ctx.beginPath(); ctx.arc(px(i), py(v), 3, 0, Math.PI * 2);
      ctx.fillStyle = '#06B6D4'; ctx.fill();
      if (i % Math.ceil(n / 6) === 0) {
        ctx.fillStyle = TEXT; ctx.font = '9px sans-serif'; ctx.textAlign = 'center';
        const lbl = labels[i].length > 7 ? labels[i].slice(0, 6) + '…' : labels[i];
        ctx.fillText(lbl, px(i), pad.top + cH + 14);
      }
    });
  }

  // ── Pie ───────────────────────────────────────────────────────────────
  private drawPie(ctx: CanvasRenderingContext2D, W: number, H: number, labels: string[], values: number[]): void {
    const legendH = Math.ceil(labels.length / 2) * 18 + 8;
    const r = Math.min(W / 2, (H - legendH)) * 0.42;
    const cx = W / 2, cy = (H - legendH) / 2;
    const total = values.reduce((a, b) => a + b, 0) || 1;
    let start = -Math.PI / 2;

    values.forEach((v, i) => {
      const slice = (v / total) * 2 * Math.PI;
      ctx.beginPath(); ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, r, start, start + slice);
      ctx.closePath();
      ctx.fillStyle = COLORS[i % COLORS.length]; ctx.fill();
      ctx.strokeStyle = BG; ctx.lineWidth = 2; ctx.stroke();
      start += slice;
    });

    // Legend
    const cols = 2;
    const itemW = W / cols;
    labels.forEach((l, i) => {
      const col = i % cols, row = Math.floor(i / cols);
      const lx = col * itemW + 12;
      const ly = H - legendH + row * 18 + 12;
      ctx.fillStyle = COLORS[i % COLORS.length];
      ctx.fillRect(lx, ly - 7, 10, 10);
      ctx.fillStyle = '#c9d1d9'; ctx.font = '10px sans-serif'; ctx.textAlign = 'left';
      const lbl = l.length > 14 ? l.slice(0, 13) + '…' : l;
      ctx.fillText(lbl, lx + 14, ly + 2);
    });
  }

  private fmt(n: number): string {
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
    return Number.isInteger(n) ? String(n) : n.toFixed(1);
  }

  onKeydown(e: KeyboardEvent): void { if (e.key === 'Enter') this.send(); }
  ngAfterViewChecked(): void { this.msgEnd?.nativeElement?.scrollIntoView({ behavior: 'smooth' }); }
  ngOnDestroy(): void {}
}
