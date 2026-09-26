"""Real HTTP round trips against the FastAPI app under uvicorn on an ephemeral loopback port, mock backend."""

import json
import threading
import time

import pytest
import requests
import uvicorn

from tokenfold.server import create_app, make_engine

REQ = {"state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}}}


def serve(api_key: str | None = None):
    """Start uvicorn on 127.0.0.1:0 in a daemon thread; return (server, base_url)."""
    app = create_app(make_engine("mock", None), api_key)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, f"http://127.0.0.1:{port}"


@pytest.fixture
def base_url():
    server, url = serve()
    yield url
    server.should_exit = True


def test_ok(base_url):
    r = requests.post(f"{base_url}/v1/tokenfold", json=REQ, timeout=5)
    assert r.status_code == 200
    assert r.json()["answers"]["q"]["type"] == "noul"


def test_typesafe_path_is_an_alias(base_url):
    r = requests.post(f"{base_url}/v1/systemone", json=REQ, timeout=5)
    assert r.status_code == 200 and r.json()["answers"]["q"]["type"] == "noul"


def test_health(base_url):
    r = requests.get(f"{base_url}/health", timeout=5)
    assert r.status_code == 200 and r.json()["backend"] == "mock"


def test_validation_is_422(base_url):
    bad = {"state": "s", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"a": None}}}}
    r = requests.post(f"{base_url}/v1/tokenfold", json=bad, timeout=5)
    assert r.status_code == 422 and "at least 2 options" in r.json()["error"]


def test_bad_json_is_422(base_url):
    r = requests.post(f"{base_url}/v1/tokenfold", data=b"{nope", timeout=5)
    assert r.status_code == 422 and "error" in r.json()


def test_unknown_path_is_404(base_url):
    r = requests.post(f"{base_url}/v1/other", json=REQ, timeout=5)
    assert r.status_code == 404 and "error" in r.json()


def test_api_key():
    server, base_url = serve(api_key="k")
    try:
        url = f"{base_url}/v1/tokenfold"
        assert requests.post(url, json=REQ, timeout=5).status_code == 401
        ok = requests.post(url, data=json.dumps(REQ), headers={"Authorization": "Bearer k"}, timeout=5)
        assert ok.status_code == 200
    finally:
        server.should_exit = True
