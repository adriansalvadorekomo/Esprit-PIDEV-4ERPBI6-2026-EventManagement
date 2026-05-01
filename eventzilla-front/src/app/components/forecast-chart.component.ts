import {
  Component, Input, OnChanges, OnDestroy,
  ElementRef, ViewChild, AfterViewInit, inject
} from '@angular/core';
import { createChart, IChartApi, ISeriesApi, LineData, Time, LineSeries } from 'lightweight-charts';
import { ThemeService } from '../services/theme.service';
import { ForecastResponse } from '../models/ml.models';

@Component({
  selector: 'app-forecast-chart',
  standalone: false,
  template: `<div #chartContainer class="lw-chart"></div>`,
  styles: [`.lw-chart { width: 100%; height: 220px; border-radius: 8px; overflow: hidden; }`]
})
export class ForecastChartComponent implements AfterViewInit, OnChanges, OnDestroy {
  @Input() data!: ForecastResponse;
  @ViewChild('chartContainer', { static: true }) container!: ElementRef<HTMLDivElement>;

  private chart?: IChartApi;
  private historySeries?: ISeriesApi<'Line'>;
  private forecastSeries?: ISeriesApi<'Line'>;
  private readonly theme = inject(ThemeService);

  ngAfterViewInit(): void {
    this.initChart();
    this.renderData();
  }

  ngOnChanges(): void {
    if (this.chart) this.renderData();
  }

  ngOnDestroy(): void {
    this.chart?.remove();
  }

  private isDark(): boolean {
    return this.theme.theme() === 'dark';
  }

  private initChart(): void {
    const dark = this.isDark();
    this.chart = createChart(this.container.nativeElement, {
      width: this.container.nativeElement.clientWidth,
      height: 220,
      layout: {
        background: { color: dark ? '#0d1117' : '#ffffff' },
        textColor: dark ? '#8b949e' : '#475569',
      },
      grid: {
        vertLines: { color: dark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)' },
        horzLines: { color: dark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)' },
      },
      rightPriceScale: { borderColor: dark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)' },
      timeScale: { borderColor: dark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)', timeVisible: true },
      handleScroll: false,
      handleScale: false,
    });

    this.historySeries = this.chart.addSeries(LineSeries, { color: '#388bfd', lineWidth: 2, title: 'History' });
    this.forecastSeries = this.chart.addSeries(LineSeries, { color: '#f778ba', lineWidth: 2, lineStyle: 1, title: 'Forecast' });
  }

  private renderData(): void {
    if (!this.chart || !this.data) return;

    const toTime = (d: string): Time => d as Time;

    const historyPoints: LineData[] = (this.data.history ?? []).map(p => ({
      time: toTime(p.date), value: p.value
    }));

    // bridge: last history point repeated as first forecast point for visual continuity
    const lastHistory = historyPoints[historyPoints.length - 1];
    const forecastPoints: LineData[] = this.data.forecast.map(p => ({
      time: toTime(p.date), value: p.value
    }));
    if (lastHistory) forecastPoints.unshift(lastHistory);

    this.historySeries!.setData(historyPoints);
    this.forecastSeries!.setData(forecastPoints);
    this.chart.timeScale().fitContent();
  }
}
