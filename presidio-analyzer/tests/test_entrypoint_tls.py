"""TLS flag behavior of entrypoint.sh under the TLS_*_FILE variables."""

import os
import stat
import subprocess
from pathlib import Path

ENTRYPOINT = Path(__file__).resolve().parents[1] / "entrypoint.sh"


def _gunicorn_args(tls_env: dict, tmp_path: Path) -> str:
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
    assert args == "-w 1 --worker-tmp-dir /dev/shm -b 0.0.0.0:3000 app:create_app()"
