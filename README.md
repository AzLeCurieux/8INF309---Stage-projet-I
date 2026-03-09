# Promotion Intelligence and Gastronomy Analytics System (PIGAS)

## Overview

The Promotion Intelligence and Gastronomy Analytics System is an automated framework designed to aggregate, classify, and analyze promotional data from heterogeneous restaurant web architectures. The system employs advanced heuristic-based discovery, Large Language Model (LLM) powered extraction, and semantic deduplication to provide a comprehensive dashboard for competitive market intelligence.

## Technical Architecture

### 1. Data Acquisition Layer
The system utilizes a multi-stage scraping pipeline to ensure high resilience against diverse web technologies:
*   **Asynchronous Fetching**: Primary data retrieval via `curl_cffi` with Chrome TLS impersonation to bypass standard bot detection.
*   **Dynamic Fallback**: Integration of Playwright (via Scrapling DynamicFetcher) for JavaScript-heavy Single Page Applications (SPAs).
*   **Heuristic Discovery**: A recursive link-scoring algorithm that identifies sub-pages with a high probability of containing promotional content based on anchor text and URL metadata.

### 2. Information Extraction & Processing
Data refinement is achieved through a probabilistic approach:
*   **LLM-Powered Extraction**: Utilization of OpenAI GPT-4o and GPT-4o-mini to transform unstructured HTML/Text into structured JSON entities.
*   **Vision Integration**: Specialized module for extracting promotional data directly from banner images using GPT-4o Vision.
*   **Semantic Deduplication**: Implementation of vector embeddings (SentenceTransformer) and cosine similarity to identify and merge redundant offers, even when textual descriptions vary.

### 3. Classification Framework
Promotions are evaluated and graded based on a standardized baseline:
*   **Quantitative Scoring**: Automated grading from A+ to D based on price-to-value heuristics.
*   **Categorization**: Keyword-based taxonomy (Solo, Family, Happy Hour, etc.) for granular data filtering.

## System Components

### Backend
*   **Framework**: Flask (Python 3.12)
*   **Database**: MySQL 8.0 for persistent relational storage.
*   **Task Management**: Asynchronous job execution with real-time log streaming via UUID-based job tracking.
*   **Scheduling**: APScheduler for periodic automated crawl cycles.

### Frontend
*   **Technology**: Server-side rendering with Jinja2 templates.
*   **Design**: Modern UI utilizing Syne and Plus Jakarta Sans typography, featuring responsive grid layouts and real-time job monitoring panels.
*   **Interactivity**: Dynamic filtering by grade and category with persistent theme state management.

### Infrastructure
*   **Containerization**: Fully Dockerized environment using Docker Compose.
*   **Security**: Nginx-based Gatekeeper proxy with HTBasic authentication.
*   **Connectivity**: Integrated support for Tailscale Funnel for secure public exposition.

## Installation and Deployment

### Prerequisites
*   Docker and Docker Compose
*   OpenAI API Key

### Configuration
Create a `.env` file in the root directory with the following parameters:
```env
DB_PASSWORD=your_secure_password
OPENAI_API_KEY=your_openai_key
SCRAPER_PROXY=optional_proxy_url
SCRAPE_INTERVAL_HOURS=6
```

### Execution
The system is managed via a centralized toolkit:
*   **Build and Start**: `./manage.sh update`
*   **Monitor Logs**: `./manage.sh logs`
*   **Database Backup**: `./manage.sh backup`

## Research and Development
This project demonstrates the application of modern AI techniques to the field of competitive intelligence within the food service industry. By automating the extraction of unstructured data, the system provides a scalable solution for real-time market monitoring.
