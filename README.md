# Promo Dashboard

Système automatisé de surveillance et classification des promotions de restaurants québécois.

Scrape les pages promotions, extrait les offres avec GPT-4o-mini, les classe par grade de valeur et les affiche dans un dashboard web.

---

## Démarrage rapide (Docker)

```bash
git clone https://github.com/AzLeCurieux/8INF309---Stage-projet-I.git
cd 8INF309---Stage-projet-I
cp .env.example .env          # puis remplis OPENAI_API_KEY dans .env
docker compose up
```

Ouvre **http://localhost:5000**

Au premier démarrage, les tables MySQL et les 4 restaurants par défaut sont créés automatiquement.

---

## Démarrage local (sans Docker)

**Prérequis :** Python 3.10+, MySQL 8

```bash
pip install -r requirements.txt
patchright install chromium      # navigateur pour les sites JavaScript

cp .env.example .env             # configurer DB_HOST, DB_PASSWORD, OPENAI_API_KEY
bash run.sh
```

---

## Configuration (`.env`)

| Variable | Défaut | Description |
|----------|--------|-------------|
| `OPENAI_API_KEY` | — | **Requis.** Clé OpenAI pour GPT-4o-mini |
| `DB_HOST` | `db` | Hôte MySQL (`127.0.0.1` en local, `db` en Docker) |
| `DB_PORT` | `3306` | Port MySQL |
| `DB_USER` | `root` | Utilisateur MySQL |
| `DB_PASSWORD` | `1234` | Mot de passe MySQL |
| `DB_NAME` | `promotions_db` | Nom de la base de données |
| `SCRAPE_INTERVAL_HOURS` | `6` | Fréquence du scraping automatique (heures) |
| `SKIP_EMBEDDING_INIT` | `1` | `1` = déduplication par texte exact (rapide) · `0` = embeddings sémantiques |

---

## Restaurants par défaut

Au premier démarrage, 4 restaurants sont pré-configurés :

| Restaurant | URL |
|------------|-----|
| Benny and Co | https://bennyandco.ca/en/promotions |
| Boston Pizza | https://www.bostonpizza.com/en/specials |
| Mike | https://toujoursmikes.ca/offres-promotions |
| Normandin | https://www.restaurantnormandin.com/fr/promotions |

D'autres restaurants peuvent être ajoutés depuis le dashboard.

---

## Fonctionnement

```
URL restaurant
    ↓
Découverte intelligente de liens (scoring par mots-clés)
    ↓
Scrapling AsyncFetcher (HTTP/curl_cffi, stealth)
    ↓ si JS-rendered
Scrapling DynamicFetcher (Playwright/Chromium, resources bloquées)
    ↓
GPT-4o-mini → extraction JSON des promotions
    ↓ si vide
GPT-4o fallback
    ↓
MySQL (déduplication + grade + catégorie)
    ↓
Dashboard Flask + Bootstrap 5
```

**Fire & forget :** `POST /crawl/<id>` retourne immédiatement un `job_id`. Le frontend poll `GET /job/<job_id>` toutes les 2 s.

---

## Système de grades

| Grade | Prix | Description |
|-------|------|-------------|
| A+ | < 7 $ | Excellente affaire |
| A | 7–12 $ | Très bonne affaire |
| B | 12–16 $ | Bonne affaire |
| C | 16–20 $ | Prix raisonnable |
| D | > 20 $ | Au-dessus de la référence |
| N/A | — | Prix non disponible |

Référence : **20 $** · Économies estimées = `max(0, 20 − prix)`

---

## API

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/` | Dashboard (liste restaurants) |
| GET | `/restaurant/<id>` | Promotions d'un restaurant |
| POST | `/crawl/<id>` | Lancer un scrape → `{job_id}` |
| POST | `/crawl_all` | Scraper tous les restaurants |
| GET | `/job/<job_id>` | Statut : `pending / running / done / error` |
| POST | `/add_restaurant` | Ajouter un restaurant |
| POST | `/delete_restaurant/<id>` | Supprimer un restaurant |
| POST | `/classify_all` | Classer les promos sans grade |
| GET | `/api/promotions` | JSON promos (`?restaurant=X&active=1`) |
| GET | `/api/restaurants` | JSON restaurants |
| GET | `/ping` | Health check |

---

## Stack technique

| Couche | Technologie |
|--------|-------------|
| Scraping HTTP | Scrapling `AsyncFetcher` (curl_cffi, TLS fingerprint stealth) |
| Scraping SPA | Scrapling `DynamicFetcher` (Playwright/patchright, Chromium) |
| LLM extraction | OpenAI GPT-4o-mini (primaire) + GPT-4o (fallback) |
| Base de données | MySQL 8 |
| Backend | Flask 3 async + Hypercorn (ASGI) |
| Scheduler | APScheduler (auto-scrape toutes les N heures) |
| Frontend | Bootstrap 5 + JavaScript vanilla |

---

## Structure

```
.
├── server.py           # Backend Flask (scraper + API + scheduler)
├── requirements.txt    # Dépendances Python
├── Dockerfile          # Image Docker
├── docker-compose.yml  # Stack complète (app + MySQL)
├── run.sh              # Lancement local
├── .env.example        # Template de configuration
└── templates/
    ├── index.html      # Dashboard liste restaurants
    ├── restaurant.html # Détail promotions
    └── about.html      # Documentation technique
```
