import { Component, OnInit } from '@angular/core';
import gsap from 'gsap';

@Component({
  selector: 'app-body',
  templateUrl: './body.component.html',
  styleUrl: './body.component.css',
  standalone: false
})
export class BodyComponent implements OnInit {
  ngOnInit(): void {
    gsap.fromTo('.body-content', { opacity: 0 }, { opacity: 1, duration: 0.5, delay: 0.1 });
  }
}
