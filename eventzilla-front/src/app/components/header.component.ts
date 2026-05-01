import { Component, OnInit, OnDestroy, inject, signal } from '@angular/core';
import { ThemeService } from '../services/theme.service';
import { AuthService } from '../services/auth.service';

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

  private onScroll = () => {
    document.querySelector('.header-nav')?.classList.toggle('scrolled', window.scrollY > 10);
  };

  ngOnInit(): void {
    window.addEventListener('scroll', this.onScroll, { passive: true });
  }

  ngOnDestroy(): void {
    window.removeEventListener('scroll', this.onScroll);
  }

  navigate(section: string): void {
    window.location.hash = section;
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
