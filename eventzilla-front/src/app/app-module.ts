import { NgModule, provideBrowserGlobalErrorListeners, provideZonelessChangeDetection } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClientModule, HTTP_INTERCEPTORS } from '@angular/common/http';
import { LucideAngularModule, BarChart2, TrendingUp, Users, Zap, Lightbulb, Settings, Rocket, Star, Brain, Search, ChevronRight, ArrowRight, Target, Award, Shield, Home, LayoutDashboard, FlaskConical, Info, Sun, Moon, LogIn, LogOut, Menu, X } from 'lucide-angular';

import { NzInputNumberModule } from 'ng-zorro-antd/input-number';
import { NzSelectModule } from 'ng-zorro-antd/select';
import { NzDatePickerModule } from 'ng-zorro-antd/date-picker';
import { NzInputModule } from 'ng-zorro-antd/input';
import { NzButtonModule } from 'ng-zorro-antd/button';
import { NzFormModule } from 'ng-zorro-antd/form';
import { NZ_I18N, en_US } from 'ng-zorro-antd/i18n';

import { AppRoutingModule } from './app-routing-module';
import { App } from './app';
import { HeaderComponent } from './components/header.component';
import { BodyComponent } from './components/body.component';
import { FooterComponent } from './components/footer.component';
import { HomeComponent } from './pages/home.component';
import { ChatbotComponent } from './components/chatbot.component';
import { ForecastChartComponent } from './components/forecast-chart.component';
import { NgrokInterceptor } from './services/ngrok.interceptor';

@NgModule({
  declarations: [
    App,
    HeaderComponent,
    BodyComponent,
    FooterComponent,
    HomeComponent,
    ChatbotComponent,
    ForecastChartComponent
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,
    CommonModule,
    FormsModule,
    HttpClientModule,
    AppRoutingModule,
    LucideAngularModule.pick({ BarChart2, TrendingUp, Users, Zap, Lightbulb, Settings, Rocket, Star, Brain, Search, ChevronRight, ArrowRight, Target, Award, Shield, Home, LayoutDashboard, FlaskConical, Info, Sun, Moon, LogIn, LogOut, Menu, X }),
    NzInputNumberModule,
    NzSelectModule,
    NzDatePickerModule,
    NzInputModule,
    NzButtonModule,
    NzFormModule,
  ],
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideZonelessChangeDetection(),
    { provide: HTTP_INTERCEPTORS, useClass: NgrokInterceptor, multi: true },
    { provide: NZ_I18N, useValue: en_US },
  ],
  bootstrap: [App]
})
export class AppModule { }
