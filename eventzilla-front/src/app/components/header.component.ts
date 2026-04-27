import { Component, OnInit, inject, signal } from '@angular/core';
import { ThemeService } from '../services/theme.service';
import { AuthService } from '../services/auth.service';
import gsap from 'gsap';

@Component({
  selector: 'app-header',
  templateUrl: './header.component.html',
  styleUrl: './header.component.css',
  standalone: false
})
export class HeaderComponent implements OnInit {
  protected readonly theme = inject(ThemeService);
  protected readonly auth  = inject(AuthService);

  showModal  = signal(false);
  email      = signal('');
  password   = signal('');
  loginError = signal('');

  ngOnInit(): void {
    gsap.from('.header-nav', { opacity: 0, y: -20, duration: 0.6, ease: 'power2.out' });
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
