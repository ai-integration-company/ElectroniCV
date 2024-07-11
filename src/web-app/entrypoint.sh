#!/bin/sh

# Start the FastAPI server
uvicorn web_app:app --host 0.0.0.0 --port 8000 --reload