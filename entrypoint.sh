#!/bin/bash
set -euo pipefail

if [ "${SCRAPER_RUN_ON_START:-0}" = "1" ]; then
  echo "Running scraper and chunker once on startup"
  python -m scraper.main
fi

if [ "${REINDEX_ON_START:-1}" = "1" ]; then
  echo "Rebuilding Elasticsearch index after scraping and chunking"
  python -m scraper.index_to_elasticsearch
fi

python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 &
# Wait for backend readiness before starting Streamlit
echo "Waiting for backend to become ready..."
if python - <<'PY'
import time
import requests

url = 'http://127.0.0.1:8000/ready'
for _ in range(60*5):
    try:
        r = requests.get(url, timeout=1)
        if r.status_code == 200:
            print('Backend ready')
            raise SystemExit(0)
    except Exception:
        pass
    time.sleep(1)
raise SystemExit(1)
PY
then
  echo "Backend ready"
else
  echo "Backend did not become ready in time"
  exit 1
fi

streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501 &

wait -n
