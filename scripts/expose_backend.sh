#!/bin/bash
set -euo pipefail

BACKEND_PORT="${BACKEND_PORT:-8000}"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"

if command -v cloudflared >/dev/null 2>&1; then
  exec cloudflared tunnel --url "http://${BACKEND_HOST}:${BACKEND_PORT}"
fi

if command -v ngrok >/dev/null 2>&1; then
  exec ngrok http "${BACKEND_PORT}"
fi

echo "Install cloudflared or ngrok, then rerun this script to expose the backend over HTTPS."
exit 1