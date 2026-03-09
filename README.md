# Promotion Intelligence and Gastronomy Analytics System (PIGAS)

## Overview
Automated framework for aggregating, classifying, and analyzing promotional data from heterogeneous restaurant web architectures. Employs LLM-based extraction and semantic deduplication for competitive market intelligence.

## Technical Specifications

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend** | Python 3.12 / Flask | Core application logic and API |
| **Database** | MySQL 8.0 | Persistent storage for restaurants and promotions |
| **Extraction** | OpenAI GPT-4o / Mini | Structured JSON extraction from HTML and Images |
| **Scraping** | curl_cffi / Playwright | Resilient data acquisition (Anti-bot bypass & JS Rendering) |
| **NLP** | SentenceTransformers | Vector embeddings for semantic deduplication |
| **Scheduling** | APScheduler | Periodic automated crawl cycles |
| **Proxy** | Nginx | Gatekeeper with HTBasic authentication |

## System Architecture

### 1. Data Acquisition
*   **Hybrid Fetching**: Synchronous `curl_cffi` for performance; asynchronous Playwright for dynamic content.
*   **Heuristic Discovery**: Link-scoring algorithm (0-100) based on anchor text and metadata to identify promo pages.

### 2. Processing Pipeline
*   **LLM Extraction**: Probabilistic transformation of unstructured text into structured JSON.
*   **Vision Module**: OCR and contextual analysis of banner images.
*   **Deduplication**: Cosine similarity check on vector embeddings to eliminate redundant entries.

## Deployment & Management

### Commands

| Action | Command |
| :--- | :--- |
| **Update/Build** | `./manage.sh update` |
| **Start Services** | `./manage.sh start` |
| **Stop Services** | `./manage.sh stop` |
| **Monitor Logs** | `./manage.sh logs` |
| **Check Status** | `./manage.sh status` |
| **DB Backup** | `./manage.sh backup` |

### Configuration (`.env`)
| Variable | Description |
| :--- | :--- |
| `DB_PASSWORD` | MySQL root password |
| `OPENAI_API_KEY` | OpenAI API authentication key |
| `SCRAPER_PROXY` | Optional HTTP/SOCKS5 proxy |
| `SCRAPE_INTERVAL_HOURS` | Frequency of automated scraping |

## Infrastructure
The system is fully containerized via **Docker Compose**, ensuring environment parity. Security is managed through an **Nginx Gatekeeper** proxy with persistent authentication, and exposition is handled via **Tailscale Funnel**.
