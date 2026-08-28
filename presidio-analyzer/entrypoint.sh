#!/bin/sh
exec gunicorn \
  -w "$WORKERS" \
  --worker-tmp-dir "${WORKER_TMP_DIR:-/dev/shm}" \
  -b "0.0.0.0:$PORT" \
  "app:create_app()"
