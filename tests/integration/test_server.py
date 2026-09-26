"""Real HTTP round trips against the stdlib server on an ephemeral loopback port, mock backend."""

import json
import threading
from http.server import ThreadingHTTPServer

import pytest
import requests

from tokenfold.server import Handler, make_engine

REQ = {"state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}}}


@pytest.fixture
def base_url():
    Handler.engine = make_engine("mock", None)
    Handler.api_key = None
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()
    Handler.api_key = None


def test_ok(base_url):
    r = requests.post(f"{base_url}/v1/tokenfold", json=REQ, timeout=5)
    assert r.status_code == 200
    assert r.json()["answers"]["q"]["type"] == "noul"


def test_typesafe_path_is_an_alias(base_url):
    r = requests.post(f"{base_url}/v1/systemone", json=REQ, timeout=5)
    assert r.status_code == 200 and r.json()["answers"]["q"]["type"] == "noul"


def test_validation_is_422(base_url):
    bad = {"state": "s", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"a": None}}}}
    r = requests.post(f"{base_url}/v1/tokenfold", json=bad, timeout=5)
    assert r.status_code == 422 and "at least 2 options" in r.json()["error"]


def test_bad_json_is_422(base_url):
    r = requests.post(f"{base_url}/v1/tokenfold", data=b"{nope", timeout=5)
    assert r.status_code == 422


def test_unknown_path_is_404(base_url):
    assert requests.post(f"{base_url}/v1/other", json=REQ, timeout=5).status_code == 404


def test_api_key(base_url):
    Handler.api_key = "k"
    url = f"{base_url}/v1/tokenfold"
    assert requests.post(url, json=REQ, timeout=5).status_code == 401
    ok = requests.post(url, data=json.dumps(REQ), headers={"Authorization": "Bearer k"}, timeout=5)
    assert ok.status_code == 200
