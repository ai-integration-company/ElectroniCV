#!/bin/sh

# Start the Telegram bot
python main.py

# Start the FastAPI server
uvicorn web_app:app --host 0.0.0.0 --port 80 --reload