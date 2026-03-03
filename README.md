# EventZella BI – Business Intelligence Platform

## Overview

This project was developed as part of the PIDEV – 4th ERPBI6 Year Engineering Program at **Esprit School of Engineering** (Academic Year 2025–2026).

EventZella BI is a Business Intelligence platform that transforms operational data from the EventZella system into strategic, data-driven insights.

EventZella, developed by Teckcatalyze, is an event management solution that allows users to:

- Manage events using an intelligent budgeting system  
- Explore, compare, and book service providers  

Although the operational system manages transactions efficiently, its data was not being fully leveraged for analytics and decision-making. This BI platform introduces a structured analytical layer to extract business value from that data.

---

## Features

- Interactive analytical dashboards  
- User behavior analysis (budgets, preferences, event types)  
- Provider performance tracking (reservations, ratings, complaints)  
- Market trend analysis  
- KPI monitoring for strategic decisions  
- Automated ETL workflows  
- Historical and predictive analytics support  

---

## Tech Stack

### Frontend
- Angular (Interactive Analytical Dashboard)
- Power BI (Data Visualization & Reporting)

### Backend
- Python (Flask)
- PostgreSQL (Data Warehouse)
- Talend (ETL)
- Apache Airflow (Scheduling)
- n8n (Automation & Alerts)

---

## Architecture

### Data Flow

```
Operational Events
        ↓
MongoDB (Source)
        ↓
Talend (ETL)
        ↓
PostgreSQL (Data Warehouse)
        ↓
Flask API
        ↓
Angular Dashboard
```

The system separates transactional processing from analytical processing to ensure:

- Clean data modeling  
- Performance optimization  
- Scalable reporting  
- Strategic insight generation  

---

## Business Context

EventZella generates daily data related to:

- User preferences and budgets  
- Provider activity and performance  
- Market trends in the Tunisian event sector  

### Problem Statement

These datasets were underutilized.

The objective of this BI system is to:

- Improve recommendation algorithms  
- Help providers enhance their services  
- Identify market opportunities  
- Support data-driven strategic decisions  

---

## Getting Started

### Prerequisites

- Python 3.10+  
- PostgreSQL   
- Angular CLI  
- Talend Open Studio  
- Apache Airflow  

---

### Installation

1. Clone the repository:

```bash
git clone https://github.com/adriansalvadorekomo/Esprit-PIDEV-4ERPBI6-2026-EventManagement.git
```

---

### Backend (Flask)

```bash
python -m venv venv
```

Activate:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run server:

```bash
flask run
```

---

### Frontend (Angular)

```bash
cd frontend
npm install
ng serve
```

---

## Contributors

- Walid Fehry  
- Emna Trabelsi  
- Hejer Mnejja  
- Karim Makni  
- Amir Jabeur  
- Adrian Salvador Ekomo Mesi Obono  

4th ERPBI6 Year Engineering Students  
Esprit School of Engineering – Tunisia  

---

## Academic Context

Developed at **Esprit School of Engineering – Tunisia**  
PIDEV – 4ERPBI6 | Academic Year 2025–2026  

This project focuses on Business Intelligence, Data Engineering, and enterprise analytics systems.

---

## Repository Description (GitHub)

Developed at Esprit School of Engineering – Tunisia  
Academic Year 2025–2026  
Main Technologies: Angular, Flask, PostgreSQL, Talend  

---

## Required Topics

- esprit-school-of-engineering  
- academic-project  
- esprit-pidev  
- 2025-2026  
- business-intelligence  
- angular  
- flask  
- postgresql  
- talend  

---

## Acknowledgments

We thank Teckcatalyze and the academic staff of Esprit School of Engineering for their guidance and support.