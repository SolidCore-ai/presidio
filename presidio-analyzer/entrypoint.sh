#!/bin/sh
# When the RD-3617 TLS_*_FILE contract is set, serve mutual TLS: present the
# leaf, verify clients against the CA (--cert-reqs 2 = ssl.CERT_REQUIRED,
# integer per gunicorn). Unset means plaintext, exactly as before. The vars
# carry file paths, never certificate material.
if [ -n "$TLS_CERT_FILE" ]; then
  set -- --certfile "$TLS_CERT_FILE" --keyfile "$TLS_KEY_FILE" --ca-certs "$TLS_CA_FILE" --cert-reqs 2
else
  set --
fi
exec gunicorn \
  -w "$WORKERS" \
  --worker-tmp-dir "${WORKER_TMP_DIR:-/dev/shm}" \
  -b "0.0.0.0:$PORT" \
  "$@" \
  "app:create_app()"
