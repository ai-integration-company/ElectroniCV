#!/bin/sh

python main.py

uvicorn web_app:app --host 0.0.0.0 --port 80 --reload