import { Component, OnInit, OnDestroy, inject, signal } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { ApiService } from '../services/api.service';
import { AuthService } from '../services/auth.service';

type Section = 'overview' | 'dashboards' | 'lab' | 'about';
const PROTECTED: Section[] = ['dashboards', 'lab'];

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrl: './home.component.css',
  standalone: false
})
export class HomeComponent implements OnInit, OnDestroy {
  private readonly api       = inject(ApiService);
  private readonly sanitizer = inject(DomSanitizer);
  protected readonly auth    = inject(AuthService);

  readonly pbiUrl     = 'https://app.powerbi.com/reportEmbed?reportId=39e494a8-2968-437f-beb2-aaa2fb9b4625&autoAuth=true&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730';
  readonly pbiUrlSafe: SafeResourceUrl = this.sanitizer.bypassSecurityTrustResourceUrl(this.pbiUrl);

  activeSection = signal<Section>('overview');

  result  = signal<number | null>(null);
  loading = signal(false);
  error   = signal<string | null>(null);

  private onHashChange = () => this.readHash();

  ngOnInit(): void {
    this.readHash();
    window.addEventListener('hashchange', this.onHashChange);
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
      this.activeSection.set(hash);
    }
  }

  isProtectedAndLocked(section: Section): boolean {
    return PROTECTED.includes(section) && !this.auth.isAdmin;
  }

  testPrediction(): void {
    this.loading.set(true);
    this.result.set(null);
    this.error.set(null);
    this.api.predict({ attendees: 100, duration: 5 }).subscribe({
      next: (res) => { this.result.set(res.prediction); this.loading.set(false); },
      error: ()    => { this.error.set('API unreachable.'); this.loading.set(false); }
    });
  }
}
