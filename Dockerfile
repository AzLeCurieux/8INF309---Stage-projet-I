# ── Promo Dashboard ───────────────────────────────────────────────────────────
# Flask + Scrapling (Playwright/patchright) + GPT-4o-mini
# Usage: docker compose up  (see docker-compose.yml)
FROM python:3.12-slim

# System libraries required by Chromium (Playwright/patchright)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl ca-certificates gnupg \
    fonts-liberation libasound2 libatk-bridge2.0-0 libatk1.0-0 \
    libatspi2.0-0 libcairo2 libcups2 libdbus-1-3 libdrm2 libgbm1 \
    libglib2.0-0 libgtk-3-0 libnspr4 libnss3 libpango-1.0-0 \
    libwayland-client0 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 \
    libxdamage1 libxext6 libxfixes3 libxkbcommon0 libxrandr2 libxrender1 \
    libxtst6 xdg-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Chromium browser for Scrapling/patchright
RUN patchright install chromium

# Copy application source
COPY . .

# Runtime defaults (overridden by docker-compose / env_file)
ENV DB_HOST=db \
    DB_PORT=3306 \
    DB_NAME=promotions_db \
    DB_USER=root \
    SKIP_EMBEDDING_INIT=1 \
    SCRAPE_INTERVAL_HOURS=6

EXPOSE 5000

CMD ["python", "server.py"]
