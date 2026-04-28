import { Component, OnInit, OnDestroy, inject, signal } from '@angular/core';
import { ThemeService } from '../services/theme.service';
import { AuthService } from '../services/auth.service';
import gsap from 'gsap';

const BAR_COUNT = 8;

@Component({
  selector: 'app-header',
  templateUrl: './header.component.html',
  styleUrl: './header.component.css',
  standalone: false
})
export class HeaderComponent implements OnInit, OnDestroy {
  protected readonly theme = inject(ThemeService);
  protected readonly auth  = inject(AuthService);

  showModal  = signal(false);
  menuOpen   = signal(false);
  email      = signal('');
  password   = signal('');
  loginError = signal('');

  readonly scrollBars = Array(BAR_COUNT).fill(0);
  activeBar = signal(-1);

  private onScroll = () => {
    document.querySelector('.header-nav')?.classList.toggle('scrolled', window.scrollY > 10);
    const doc = document.documentElement;
    const pct = doc.scrollTop / (doc.scrollHeight - doc.clientHeight);
    this.activeBar.set(Math.floor(pct * BAR_COUNT));
  };

  ngOnInit(): void {
    gsap.from('.header-nav', { opacity: 0, y: -16, duration: 0.6, ease: 'power3.out', delay: 0.1 });
    gsap.from('.logo',       { opacity: 0, x: -12, duration: 0.5, ease: 'power2.out', delay: 0.25 });
    gsap.from('.nav-link',   { opacity: 0, y: -8, stagger: 0.07, duration: 0.4, ease: 'power2.out', delay: 0.3 });
    gsap.from('.header-actions > *', { opacity: 0, x: 12, stagger: 0.06, duration: 0.4, ease: 'power2.out', delay: 0.35 });
    window.addEventListener('scroll', this.onScroll, { passive: true });
  }

  ngOnDestroy(): void {
    window.removeEventListener('scroll', this.onScroll);
  }

  openModal():  void { this.showModal.set(true);  this.loginError.set(''); }
  closeModal(): void { this.showModal.set(false); this.email.set(''); this.password.set(''); this.loginError.set(''); }

  submit(): void {
    const err = this.auth.signIn(this.email(), this.password());
    if (err) { this.loginError.set(err); }
    else      { this.closeModal(); }
  }

  onKeydown(e: KeyboardEvent): void { if (e.key === 'Enter') this.submit(); }
}
