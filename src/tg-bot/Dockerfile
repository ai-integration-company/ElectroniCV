FROM python:3.9-slim AS builder

WORKDIR /tg-bot

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y poppler-utils

RUN pip install --no-cache-dir -r requirements.txt

FROM builder AS stage

WORKDIR /tg-bot

COPY . .

EXPOSE 80

CMD uvicorn web_app:app --reload --host 0.0.0.0 --port 80 & python main.py
