import { Component, signal, inject, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { ApiService, ChatResponse } from '../services/api.service';

interface Message {
  role: 'user' | 'bot';
  text: string;
  data?: Record<string, unknown>[];
  type?: string;
  loading?: boolean;
}

@Component({
  selector: 'app-chatbot',
  templateUrl: './chatbot.component.html',
  styleUrl: './chatbot.component.css',
  standalone: false
})
export class ChatbotComponent implements AfterViewChecked {
  @ViewChild('msgEnd') private msgEnd!: ElementRef;

  private readonly api = inject(ApiService);

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

  ask(question: string): void {
    this.input.set(question);
    this.send();
  }

  send(): void {
    const text = this.input().trim();
    if (!text) return;

    this.messages.update(m => [...m, { role: 'user', text }]);
    this.input.set('');
    this.messages.update(m => [...m, { role: 'bot', text: '', loading: true }]);

    this.api.chatbot(text).subscribe({
      next: (res: ChatResponse) => {
        this.messages.update(msgs => {
          const updated = [...msgs];
          updated[updated.length - 1] = {
            role: 'bot',
            text: res.reply,
            data: res.data?.length ? res.data.slice(0, 20) : undefined,
            type: res.type,
            loading: false
          };
          return updated;
        });
      },
      error: () => {
        this.messages.update(msgs => {
          const updated = [...msgs];
          updated[updated.length - 1] = {
            role: 'bot',
            text: 'Sorry, I could not process your request.',
            loading: false
          };
          return updated;
        });
      }
    });
  }

  onKeydown(e: KeyboardEvent): void { if (e.key === 'Enter') this.send(); }

  ngAfterViewChecked(): void {
    this.msgEnd?.nativeElement?.scrollIntoView({ behavior: 'smooth' });
  }
}
