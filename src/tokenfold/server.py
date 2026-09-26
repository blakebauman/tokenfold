"""Minimal HTTP server exposing POST /v1/tokenfold. Stdlib only; swap for FastAPI when you
have it — Engine/parse_request are framework-agnostic.
  python -m tokenfold.server --backend mock --port 8080
  python -m tokenfold.server --backend vllm --calibration cal.json
  python -m tokenfold.server --backend llama   # $TOKENFOLD_LLAMA_MODEL, see backends/llama_cpp.py"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .backends import LlamaCppBackend, MockBackend, VLLMLogprobBackend
from .calibration import Calibrator
from .engine import Engine
from .schema import ValidationError, parse_request

BACKENDS = {"mock": MockBackend, "vllm": VLLMLogprobBackend, "llama": LlamaCppBackend}
# `/v1/systemone` is TypeSafe's own path (not this project's old name): serving it too lets any
# TypeSafe client, such as Felix's `typesafe` decision provider, use tokenfold by changing only its
# base URL.
PATHS = ("/v1/tokenfold", "/v1/systemone")


def make_engine(backend: str, calibration: str | None) -> Engine:
    be = BACKENDS[backend]()
    cal = Calibrator.load(calibration) if calibration and os.path.exists(calibration) else None
    return Engine(be, cal)


class Handler(BaseHTTPRequestHandler):
    engine: Engine  # set by main() before serving
    api_key: str | None = None

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path not in PATHS:
            return self._send(404, {"error": "not found"})
        if self.api_key and self.headers.get("Authorization") != f"Bearer {self.api_key}":
            return self._send(401, {"error": "missing or invalid API key"})
        try:
            raw = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            req = parse_request(raw)
        except (json.JSONDecodeError, ValidationError) as e:
            return self._send(422, {"error": str(e)})
        try:
            self._send(200, self.engine.evaluate(req))
        except Exception as e:  # backend failure
            self._send(529, {"error": f"backend error: {e}"})

    def log_message(self, format: str, *args: object) -> None:  # quieter
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="mock")
    ap.add_argument("--calibration", default="calibration.json")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--api-key", default=os.environ.get("TOKENFOLD_API_KEY"))
    a = ap.parse_args()
    Handler.engine = make_engine(a.backend, a.calibration)
    Handler.api_key = a.api_key
    print(f"tokenfold listening on :{a.port} backend={Handler.engine.backend.name} T={Handler.engine.cal.T}")
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
