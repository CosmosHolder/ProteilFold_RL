"""
api/main.py
ProteinFold-RL — FastAPI application entry point.

Responsibilities
----------------
- Add project root to sys.path so all existing imports work unchanged.
- Load the model at startup via lifespan (not on first request).
- Register all route routers.
- Configure CORS for the Gradio frontend on port 7860.
- Register global exception handlers with structured ErrorResponse.
- Expose /docs (Swagger UI) and /redoc.

Run locally
-----------
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

HuggingFace Spaces
------------------
    CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]

Author : ProteinFold-RL team
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

# ── Add project root to path BEFORE any local imports ─────────
# This ensures `from env.fold_env import FoldEnv` etc. work regardless
# of the working directory from which uvicorn is launched.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.model_manager import get_model_manager, DEFAULT_CHECKPOINT
from api.routes import health, fold, results
from api.schemas import ErrorResponse

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("proteinfold.main")

# ── CORS origins ───────────────────────────────────────────────
# Development: Gradio on 7860, any localhost port.
# Production: HuggingFace Spaces URL (set via env var).
_HF_SPACE_URL = os.getenv("HF_SPACE_URL", "")   # e.g. https://user-name.hf.space

ALLOWED_ORIGINS = [
    "http://localhost:7860",
    "http://127.0.0.1:7860",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:5173",
    "http://localhost:63342",  # PyCharm built-in server
    "http://127.0.0.1:63342",
    "null",
]
if _HF_SPACE_URL:
    ALLOWED_ORIGINS.append(_HF_SPACE_URL)


# ── Lifespan ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model at startup so the first request is never slow.
    Graceful: if the checkpoint is missing, the server still boots
    (useful for HuggingFace Spaces cold starts before uploading weights).
    """
    logger.info("=" * 60)
    logger.info("ProteinFold-RL API — Starting up")
    logger.info("Checkpoint: %s", DEFAULT_CHECKPOINT)
    logger.info("=" * 60)

    mm = get_model_manager(DEFAULT_CHECKPOINT)
    try:
        mm.load()
        logger.info("Model loaded successfully.")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Model load failed at startup (%s). "
            "Server will start anyway — /fold will return 503 "
            "until a valid checkpoint is present.",
            exc,
        )

    yield  # <-- server runs here

    logger.info("ProteinFold-RL API — Shutting down.")


# ── App instance ───────────────────────────────────────────────

app = FastAPI(
    title="ProteinFold-RL API",
    description=(
        "**ProteinFold-RL** — An RL agent that discovers *how* proteins fold, "
        "not just where they end up.\n\n"
        "_AlphaFold shows the destination. We discover the journey._\n\n"
        "## Endpoints\n"
        "| Endpoint | Description |\n"
        "|----------|-------------|\n"
        "| `POST /fold` | Run the folding agent on a protein |\n"
        "| `GET /results` | Full training log (all episodes) |\n"
        "| `GET /best-episode` | Best recorded trajectory |\n"
        "| `GET /compare` | Trained agent vs random baseline |\n"
        "| `GET /health` | Liveness/readiness probe |"
    ),
    version="2.0.0",
    contact={
        "name": "ProteinFold-RL",
        "url": "https://github.com/proteinfold-rl",
    },
    license_info={"name": "MIT"},
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── CORS ───────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.hf\.space",   # any HF Space
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── Global exception handlers ──────────────────────────────────

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            detail=f"Route not found: {request.url.path}",
            code="NOT_FOUND",
        ).model_dump(),
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="An unexpected server error occurred.",
            code="INTERNAL_SERVER_ERROR",
        ).model_dump(),
    )


# ── Routers ────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(fold.router)
app.include_router(results.router)


# ── Root ───────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return {
        "project": "ProteinFold-RL",
        "tagline": "AlphaFold shows the destination. We discover the journey.",
        "docs": "/docs",
        "health": "/health",
        "version": "2.0.0",
    }


# ── Dev entry point ────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )