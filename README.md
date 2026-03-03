# EventZella BI – Business Intelligence Platform

## Overview

This project was developed as part of the PIDEV – 4rd Year Engineering Program at **Esprit School of Engineering** (Academic Year 2025–2026).

EventZella BI is a Business Intelligence platform designed to transform raw operational data from the EventZella event management system into strategic, data-driven insights.

EventZella, developed by Teckcatalyze, is an event management solution that helps users:

- Manage events using an intelligent budgeting system  
- Explore, compare, and book service providers  

Although the operational system efficiently handles transactions, valuable data remains underexploited. This BI project introduces a complete analytical layer to unlock business value from that data.

---

## Features

- Interactive dashboards and dynamic reports  
- User behavior analytics (preferences, budgets, event types)  
- Provider performance analysis (bookings, ratings, complaints)  
- Market trend analysis for the Tunisian event industry  
- KPI monitoring for strategic decision-making  
- Automated ETL data pipelines  
- Historical and trend-based analysis  
- Data-driven support for recommendation algorithm optimization  

---

## Tech Stack

### Frontend
- Angular (Interactive Dashboard)
- Power BI (Data Visualization & Reporting)

### Backend
- Node.js
- MongoDB (Operational Database)
- PostgreSQL (Data Warehouse)
- Talend (ETL)
- Apache Airflow (Workflow Orchestration)
- n8n (Automation & Alerts)

---

## Architecture

### Existing Operational System (Client Architecture)

- Frontend: React + React Native + Expo  
- Backend: Node.js + Express.js  
- Database: MongoDB + Mongoose  
- Architecture Pattern: 3-Tier MVC  

The current system is optimized for transactional operations but lacks analytical capabilities. Data is siloed inside MongoDB, resulting in reactive and manual reporting.

### BI Architecture Overview

```
Events & Transactions
        ↓
Operational Database (MongoDB)
        ↓
ETL Process (Talend)
        ↓
Data Warehouse (PostgreSQL)
        ↓
Analytics Layer (Power BI / Angular Dashboard)
        ↓
Strategic Intelligence & Smart Decisions
```

The BI layer introduces:

- Structured data transformation  
- Centralized analytical storage  
- Automated workflows  
- Historical data modeling  
- Real-time dashboards  

---

## Business Context

EventZella generates valuable daily data related to:

- User behavior (preferences, budgets, event types)  
- Provider performance (reservations, ratings, complaints)  
- Market dynamics in the Tunisian event sector  

### Problem Statement

These datasets are currently underexploited.

The company requires a Business Intelligence system to:

- Improve the event suggestion algorithm  
- Help providers optimize their services  
- Identify new market opportunities  
- Enable strategic, data-driven decision-making  

---

## Getting Started

### Prerequisites
 
- PostgreSQL  
- Talend Open Studio  
- Apache Airflow  
- Power BI Desktop  
- Angular CLI  

### Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/eventzella-bi.git
```

2. Install backend dependencies:

```bash
cd backend
npm install
```

3. Configure environment variables (`.env`)

4. Run the backend server:

```bash
npm start
```

5. Start the Angular dashboard:

```bash
cd frontend
npm install
ng serve
```

6. Configure ETL jobs in Talend and schedule workflows using Airflow.

---

## Contributors

- [Walid Fehry]
- [Emna Trabelsi]
- [Hejer Mnejja]
- [Karim Makni]
- [Amir Jabeur]
- [Adrian Salvador Ekomo Mesi Obono]

4rd Year Engineering Students  
Esprit School of Engineering – Tunisia  

---

## Academic Context

Developed at **Esprit School of Engineering – Tunisia**  
PIDEV – 4ERPBI6 | Academic Year 2025–2026  

This project was completed within the framework of the PIDEV academic module and focuses on Business Intelligence, Data Engineering, and enterprise decision systems.

---

## Repository Description (GitHub)

Developed at Esprit School of Engineering – Tunisia  
Academic Year 2025–2026  
Main Technologies: Power BI, Talend, PostgreSQL, Angular, Node.js  

---

## Required Topics (GitHub Tags)

- esprit-school-of-engineering  
- academic-project  
- esprit-pidev  
- 2025-2026  
- business-intelligence  
- power-bi  
- talend  
- postgresql  
- angular  
- nodejs  

---

## Acknowledgments

We thank:

- Teckcatalyze for providing the EventZella business case  
- Our academic supervisors at Esprit School of Engineering  
- The PIDEV program for promoting practical, industry-oriented learning  
