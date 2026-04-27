import { NgModule, provideBrowserGlobalErrorListeners, provideZonelessChangeDetection } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';

import { AppRoutingModule } from './app-routing-module';
import { App } from './app';
import { HeaderComponent } from './components/header.component';
import { BodyComponent } from './components/body.component';
import { FooterComponent } from './components/footer.component';
import { HomeComponent } from './pages/home.component';
import { ChatbotComponent } from './components/chatbot.component';

@NgModule({
  declarations: [
    App,
    HeaderComponent,
    BodyComponent,
    FooterComponent,
    HomeComponent,
    ChatbotComponent
  ],
  imports: [
    BrowserModule,
    CommonModule,
    HttpClientModule,
    AppRoutingModule
  ],
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideZonelessChangeDetection()
  ],
  bootstrap: [App]
})
export class AppModule { }
