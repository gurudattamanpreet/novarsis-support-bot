#!/usr/bin/env bash
PORT="${PORT:-8000}"
python -m uvicorn novarsis_app3:app --host 0.0.0.0 --port $PORT --proxy-headers
