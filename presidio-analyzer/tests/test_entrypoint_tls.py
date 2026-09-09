"""TLS flag behavior of entrypoint.sh under the TLS_*_FILE variables."""

import http.client
import os
import socket
import stat
import subprocess
import time
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[1] / "entrypoint.sh"


def _gunicorn_args(tls_env: dict, tmp_path: Path, extra_env: dict | None = None) -> str:
    """Run entrypoint.sh with a stub gunicorn; return the args it received."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    args_file = tmp_path / "gunicorn_args"
    stub = bin_dir / "gunicorn"
    stub.write_text(f'#!/bin/sh\necho "$@" > "{args_file}"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "WORKERS": "1",
        "PORT": "3000",
        **tls_env,
        **(extra_env or {}),
    }
    subprocess.run([str(ENTRYPOINT)], env=env, check=True, timeout=30)
    return args_file.read_text().strip()


def test_gunicorn_requires_client_certs_when_tls_env_is_set(tmp_path):
    """With TLS_*_FILE set, gunicorn serves TLS and requires client certs."""
    args = _gunicorn_args(
        {
            "TLS_CERT_FILE": "/tls/tls.crt",
            "TLS_KEY_FILE": "/tls/tls.key",
            "TLS_CA_FILE": "/tls/ca.crt",
        },
        tmp_path,
    )
    assert "--certfile /tls/tls.crt" in args
    assert "--keyfile /tls/tls.key" in args
    assert "--ca-certs /tls/ca.crt" in args
    # integer VerifyMode: 2 = ssl.CERT_REQUIRED ("require" is illegal)
    assert "--cert-reqs 2" in args


def test_gunicorn_args_are_unchanged_without_tls_env(tmp_path):
    """Without TLS_*_FILE, the gunicorn invocation matches today's exactly."""
    args = _gunicorn_args({}, tmp_path)
    assert args == (
        "-w 1 --worker-class gthread --threads 1 --keep-alive 75"
        " --worker-tmp-dir /dev/shm -b 0.0.0.0:3000 app:create_app()"
    )


def test_worker_class_supports_keep_alive_on_both_paths(tmp_path):
    """Both TLS and plaintext invocations run a keep-alive-capable worker.

    The default sync worker closes every connection after its response, which
    silently costs clients a full TLS handshake per request no matter how they
    pool connections. The worker class is therefore load-bearing for the
    service's hottest path, not a tuning preference.
    """
    for tls_env in (
        {},
        {
            "TLS_CERT_FILE": "/tls/tls.crt",
            "TLS_KEY_FILE": "/tls/tls.key",
            "TLS_CA_FILE": "/tls/ca.crt",
        },
    ):
        args = _gunicorn_args(tls_env, tmp_path)
        assert "--worker-class gthread" in args
        assert "--threads 1" in args
        assert "--keep-alive 75" in args


def test_threads_and_keep_alive_are_operator_tunable(tmp_path):
    """THREADS and KEEP_ALIVE override the defaults, mirroring WORKERS."""
    args = _gunicorn_args({}, tmp_path, extra_env={"THREADS": "4", "KEEP_ALIVE": "120"})
    assert "--threads 4" in args
    assert "--keep-alive 120" in args


def test_a_partial_tls_env_set_stops_startup(tmp_path):
    """Setting any TLS_*_FILE variable requires all three.

    The quadrant that matters: key and CA without the cert previously fell
    through to the plaintext branch and served unencrypted while the
    operator believed TLS was configured.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "gunicorn"
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    base_env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "WORKERS": "1",
        "PORT": "3000",
    }
    partial_sets = [
        {"TLS_CERT_FILE": "/tls/tls.crt"},
        {"TLS_KEY_FILE": "/tls/tls.key", "TLS_CA_FILE": "/tls/ca.crt"},
        {"TLS_CERT_FILE": "/tls/tls.crt", "TLS_KEY_FILE": "/tls/tls.key"},
    ]
    for tls_env in partial_sets:
        result = subprocess.run(
            [str(ENTRYPOINT)],
            env={**base_env, **tls_env},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0, tls_env
        assert "all-or-none" in result.stderr
        for var in {"TLS_CERT_FILE", "TLS_KEY_FILE", "TLS_CA_FILE"} - set(tls_env):
            assert var in result.stderr


def test_served_connections_are_reused_across_requests(tmp_path):
    """Two requests on one socket reach a real gunicorn booted by entrypoint.sh.

    Runs the actual entrypoint against a trivial WSGI app (gunicorn resolves
    ``app:create_app()`` from the working directory), then asserts the server
    holds the connection open between requests instead of answering
    ``Connection: close`` — the regression this file exists to prevent.
    """
    pytest.importorskip("gunicorn")
    (tmp_path / "app.py").write_text(
        "def create_app():\n"
        "    def app(environ, start_response):\n"
        "        start_response('200 OK', [('Content-Type', 'text/plain')])\n"
        "        return [b'ok']\n"
        "    return app\n"
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    env = {
        "PATH": os.environ["PATH"],
        "WORKERS": "1",
        "PORT": str(port),
        # the image default /dev/shm only exists on Linux
        "WORKER_TMP_DIR": str(tmp_path),
    }
    log_path = tmp_path / "gunicorn.log"
    with log_path.open("wb") as log:
        server = subprocess.Popen(
            [str(ENTRYPOINT)],
            cwd=tmp_path,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if server.poll() is not None:
                pytest.fail(f"gunicorn exited during boot:\n{log_path.read_text()}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.fail(f"gunicorn never started listening:\n{log_path.read_text()}")

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        # Two layers of evidence, because http.client reconnects transparently
        # after a server-side close: the Connection header states the server's
        # contract, and the socket identity proves actual reuse — a reconnect
        # creates a new socket object with a new ephemeral local port.
        conn.request("GET", "/")
        response = conn.getresponse()
        assert response.status == 200
        assert response.getheader("Connection") != "close"
        response.read()
        first_socket = conn.sock
        first_local_addr = conn.sock.getsockname()

        conn.request("GET", "/")
        response = conn.getresponse()
        assert response.status == 200
        assert response.getheader("Connection") != "close"
        response.read()
        assert conn.sock is first_socket
        assert conn.sock.getsockname() == first_local_addr
        conn.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
