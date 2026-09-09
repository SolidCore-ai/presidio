#!/bin/sh
# When the TLS_*_FILE variables are set, serve mutual TLS: present the
# leaf, verify clients against the CA (--cert-reqs 2 = ssl.CERT_REQUIRED,
# integer per gunicorn). Unset means plaintext, exactly as before. The vars
# carry file paths, never certificate material.
#
# The three variables are all-or-none: setting any one requires all three,
# and a partial set stops startup naming what is missing. In particular,
# TLS_KEY_FILE and TLS_CA_FILE without TLS_CERT_FILE would otherwise serve
# plaintext while the operator believes TLS is configured.
if [ -n "${TLS_CERT_FILE}${TLS_KEY_FILE}${TLS_CA_FILE}" ]; then
  missing=""
  [ -n "$TLS_CERT_FILE" ] || missing="$missing TLS_CERT_FILE"
  [ -n "$TLS_KEY_FILE" ] || missing="$missing TLS_KEY_FILE"
  [ -n "$TLS_CA_FILE" ] || missing="$missing TLS_CA_FILE"
  if [ -n "$missing" ]; then
    echo "entrypoint.sh: TLS_CERT_FILE, TLS_KEY_FILE, and TLS_CA_FILE are all-or-none; missing:$missing" >&2
    exit 1
  fi
fi
if [ -n "$TLS_CERT_FILE" ]; then
  set -- --certfile "$TLS_CERT_FILE" --keyfile "$TLS_KEY_FILE" --ca-certs "$TLS_CA_FILE" --cert-reqs 2
else
  set --
fi
# gthread rather than the default sync worker class: sync workers close
# every connection after the response, which forces clients back through a
# full TLS handshake per request on this service's hottest path. gthread
# honors keep-alive; with THREADS at its default of 1 each worker still
# handles one request at a time, so concurrency semantics are unchanged.
# KEEP_ALIVE stays above typical client connection-pool idle times.
exec gunicorn \
  -w "$WORKERS" \
  --worker-class gthread \
  --threads "${THREADS:-1}" \
  --keep-alive "${KEEP_ALIVE:-75}" \
  --worker-tmp-dir "${WORKER_TMP_DIR:-/dev/shm}" \
  -b "0.0.0.0:$PORT" \
  "$@" \
  "app:create_app()"
