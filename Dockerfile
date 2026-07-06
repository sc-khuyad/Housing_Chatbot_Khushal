FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    PYTHONPATH=/app \
    BACKEND_URL=http://localhost:8000/chat \
    SCRAPER_RUN_ON_START=0 \
    REINDEX_ON_START=0 

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libasound2 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libsm6 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    libxrender1 \
    libxshmfence1 \
    libxtst6 \
    wget \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt && playwright install --with-deps chromium

COPY . .

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000 8501

CMD ["/app/entrypoint.sh"]
