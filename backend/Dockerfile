FROM python:3.13.3-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
  postgresql-client \
  libpq-dev \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
