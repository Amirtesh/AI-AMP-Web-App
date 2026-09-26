"""AI-AMP model suite — FastAPI backend (no frontend in this pass).

Run:  uvicorn backend.main:app --host 0.0.0.0 --port 8000
(from Web-App/, with the ai-amp-backend conda env active)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse

from . import config
from .logging_setup import setup_logging
from .model_loader import build_state
from .routers import generate as generate_router
from .routers import predict as predict_router
from .routers import stats as stats_router

logger = setup_logging()

DESCRIPTION = f"""AI-AMP peptide model suite: Model1 spectrum classifier
(Gram+/Gram-/antifungal), Model2 AMP/non-AMP gatekeeper, and Stage 1 / Stage 2
peptide generators.

Full project download: {config.GITHUB_REPO_URL}
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load everything once at startup; refuse to start on any failure.
    try:
        app.state.app_state = build_state()
    except Exception as exc:
        logger.error(f"FATAL: model loading failed, refusing to start: {exc}")
        raise
    logger.info(
        "AI-AMP backend started: model1 (3 boosters) + model2 (XGBClassifier) "
        "+ stage1/stage2 generators + shared ESM2 loaded"
    )
    yield


app = FastAPI(
    title="AI-AMP Model Suite API",
    description=DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(predict_router.router)
app.include_router(generate_router.router)
app.include_router(stats_router.router)

FRONTEND_DIR = config.WEBAPP_DIR / "frontend"
# No-store so browsers never stick on a stale console bundle.
_NO_CACHE = {"Cache-Control": "no-store"}


@app.get(
    "/app",
    tags=["frontend"],
    summary="AI-AMP web console",
    response_class=FileResponse,
)
def serve_frontend():
    return FileResponse(
        str(FRONTEND_DIR / "index.html"),
        media_type="text/html",
        headers=_NO_CACHE,
    )


@app.get("/app/styles.css", tags=["frontend"], include_in_schema=False)
def serve_css():
    return FileResponse(
        str(FRONTEND_DIR / "styles.css"),
        media_type="text/css",
        headers=_NO_CACHE,
    )


@app.get("/app/app.js", tags=["frontend"], include_in_schema=False)
def serve_js():
    return FileResponse(
        str(FRONTEND_DIR / "app.js"),
        media_type="application/javascript",
        headers=_NO_CACHE,
    )


@app.get(
    "/",
    tags=["root"],
    summary="Service info and full-download link",
)
def root():
    return {
        "service": "AI-AMP Model Suite API",
        "version": "1.0.0",
        "models": {
            "model1": "spectrum classifier (gram_positive / gram_negative / fungal)",
            "model2": "AMP / non-AMP gatekeeper",
            "generative_stage1": "general-peptide model (NOT AMP-validated)",
            "generative_stage2": "AMP-fine-tuned model (validated deliverable)",
        },
        "endpoints": {
            "predict_model1": "POST /predict/model1",
            "predict_model2": "POST /predict/model2",
            "predict_combined": "POST /predict/combined (preferred when both scores are wanted)",
            "generate_stage1": "POST /generate/stage1",
            "generate_stage2": "POST /generate/stage2",
            "stats": "GET /stats (+ /stats/images/{name}, /stats/files/{name})",
        },
        "github_repository": config.GITHUB_REPO_URL,
        "docs": "/docs",
    }
