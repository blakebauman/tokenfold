"""FastAPI server exposing POST /v1/tokenfold. Engine/parse_request stay framework-agnostic.
tokenfold-server --backend mock --port 8080
tokenfold-server --backend vllm --calibration cal.json
tokenfold-server --backend llama   # $TOKENFOLD_LLAMA_MODEL, see backends/llama_cpp.py"""

from __future__ import annotations

import argparse
import json
import os

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

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


def create_app(engine: Engine, api_key: str | None = None) -> FastAPI:
    """Build the app around one shared engine. Every error is `{"error": str}` with the TypeSafe-style
    status codes: 401 bad key, 404 unknown path, 422 bad request, 529 backend failure."""
    app = FastAPI(title="tokenfold", docs_url=None, redoc_url=None, openapi_url=None)

    # Starlette's base class also covers the router's own 404/405, so those get the same shape.
    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True, "backend": engine.backend.name}

    async def evaluate(request: Request) -> dict:
        if api_key and request.headers.get("Authorization") != f"Bearer {api_key}":
            raise HTTPException(401, "missing or invalid API key")
        try:
            req = parse_request(json.loads(await request.body() or b"{}"))
        except (json.JSONDecodeError, ValidationError) as e:
            raise HTTPException(422, str(e)) from e
        try:
            # Backends block (llama holds a lock, vLLM uses requests), so keep the event loop free.
            return await run_in_threadpool(engine.evaluate, req)
        except Exception as e:  # backend failure
            raise HTTPException(529, f"backend error: {e}") from e

    for path in PATHS:
        app.post(path)(evaluate)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="mock")
    ap.add_argument("--calibration", default="calibration.json")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--api-key", default=os.environ.get("TOKENFOLD_API_KEY"))
    a = ap.parse_args()
    engine = make_engine(a.backend, a.calibration)
    print(f"tokenfold listening on :{a.port} backend={engine.backend.name} T={engine.cal.T}")
    uvicorn.run(create_app(engine, a.api_key), host="0.0.0.0", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
