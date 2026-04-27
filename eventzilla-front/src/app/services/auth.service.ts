import { Injectable, signal } from '@angular/core';

const CREDENTIALS = { email: 'karimmakni14@gmail.com', password: '12345678', role: 'admin', name: 'Admin' };

@Injectable({ providedIn: 'root' })
export class AuthService {
  readonly isLoggedIn = signal(localStorage.getItem('eventzilla_logged_in') === 'true');
  readonly userName   = signal(localStorage.getItem('eventzilla_user_email') ?? '');
  readonly role       = signal(localStorage.getItem('eventzilla_role') ?? '');

  /** Returns null on success, error string on failure */
  signIn(email: string, password: string): string | null {
    if (email === CREDENTIALS.email && password === CREDENTIALS.password) {
      localStorage.setItem('eventzilla_logged_in', 'true');
      localStorage.setItem('eventzilla_user_email', CREDENTIALS.email);
      localStorage.setItem('eventzilla_role', CREDENTIALS.role);
      this.isLoggedIn.set(true);
      this.userName.set(CREDENTIALS.name);
      this.role.set(CREDENTIALS.role);
      return null;
    }
    return 'Invalid email or password.';
  }

  signOut(): void {
    localStorage.removeItem('eventzilla_logged_in');
    localStorage.removeItem('eventzilla_user_email');
    localStorage.removeItem('eventzilla_role');
    this.isLoggedIn.set(false);
    this.userName.set('');
    this.role.set('');
  }

  get isAdmin(): boolean { return this.role() === 'admin'; }
}
