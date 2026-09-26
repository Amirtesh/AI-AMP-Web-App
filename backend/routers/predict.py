"""POST /predict/model1, /predict/model2, /predict/combined."""

import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .. import config
from ..esm2 import embed_esm2
from ..inference import predict_model1, predict_model2
from ..logging_setup import log_request, setup_logging
from ..rate_limit import check_predict_rate
from ..validation import validate_predict_sequences, validation_error_detail

logger = setup_logging()
router = APIRouter(prefix="/predict", tags=["predict"])


class PredictRequest(BaseModel):
    sequences: list = Field(
        ...,
        min_length=1,
        max_length=config.MAX_SEQUENCES,
        description=f"1-{config.MAX_SEQUENCES} peptide sequences (20 standard amino acids only)",
        examples=[["GLFDIVKKVVGALGSL", "GIGDPVTCLKSGAICHPVFCPRRYK"]],
    )


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _reject(endpoint: str, client: str, started: float, errors: list):
    latency_ms = (time.perf_counter() - started) * 1000.0
    reason = "; ".join(
        f"[{e['index']}] {e['reason']}" for e in errors[:5]
    )
    if len(errors) > 5:
        reason += f" (+{len(errors) - 5} more)"
    log_request(
        logger, endpoint=endpoint, client=client,
        n_sequences=len(errors), models=[], latency_ms=latency_ms,
        rejection=reason,
    )
    raise HTTPException(status_code=422, detail=validation_error_detail(errors))


@router.post(
    "/model1",
    dependencies=[Depends(check_predict_rate)],
    summary="Model1 spectrum prediction (Gram+/Gram-/fungal)",
    description=(
        "Multi-label spectrum classifier: gram_positive and fungal heads use "
        "34 biophysical features, the gram_negative head uses one shared "
        "ESM2 embedding batch. Thresholds come from model_artifacts/config.json "
        "(gp=0.50, gn=0.52, fungal=0.48). Model1 enforces no training max "
        "length in its inference code — only the 20-AA alphabet and the "
        f"{config.MAX_SEQUENCES}-sequence batch cap apply here."
    ),
)
def predict_m1(body: PredictRequest, request: Request):
    started = time.perf_counter()
    client = _client(request)
    clean, errors = validate_predict_sequences(
        body.sequences, enforce_length_range=False
    )
    if errors:
        _reject("/predict/model1", client, started, errors)
    try:
        rows, table, thresholds = predict_model1(request.app.state.app_state.model1, clean)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc)})
    latency_ms = (time.perf_counter() - started) * 1000.0
    log_request(logger, endpoint="/predict/model1", client=client,
                n_sequences=len(clean), models=["model1"], latency_ms=latency_ms)
    return {
        "results": rows,
        "table": table,
        "table_format": "csv",
        "thresholds": thresholds,
        "n_sequences": len(rows),
    }


@router.post(
    "/model2",
    dependencies=[Depends(check_predict_rate)],
    summary="Model2 AMP/non-AMP gatekeeper",
    description=(
        "Binary AMP/non-AMP classifier on shared ESM2 embeddings "
        "(XGBClassifier, threshold 0.5 from ablation_results_with_thresholds.csv, "
        f"esm2 row). Validated length range {config.MODEL2_MIN_LEN}-"
        f"{config.MODEL2_MAX_LEN}aa is enforced; sequences under "
        f"{config.MODEL2_SHORT_FLAG_THRESHOLD}aa carry an evidenced "
        "low-confidence note."
    ),
)
def predict_m2(body: PredictRequest, request: Request):
    started = time.perf_counter()
    client = _client(request)
    clean, errors = validate_predict_sequences(
        body.sequences, enforce_length_range=True
    )
    if errors:
        _reject("/predict/model2", client, started, errors)
    rows, table, threshold = predict_model2(request.app.state.app_state.model2, clean)
    latency_ms = (time.perf_counter() - started) * 1000.0
    log_request(logger, endpoint="/predict/model2", client=client,
                n_sequences=len(clean), models=["model2"], latency_ms=latency_ms)
    state = request.app.state.app_state.model2
    return {
        "results": rows,
        "table": table,
        "table_format": "csv",
        "threshold": threshold,
        "validation_metrics": state.metrics,
        "n_sequences": len(rows),
    }


@router.post(
    "/combined",
    dependencies=[Depends(check_predict_rate)],
    summary="Model1 + Model2 in one call (use this when both scores are wanted)",
    description=(
        "Runs embed_esm2 ONCE on the input batch and feeds the same embedding "
        "matrix to both Model1 and Model2. This is the endpoint the frontend "
        "should use when both scores are wanted — do NOT call /predict/model1 "
        "and /predict/model2 back to back for the same sequences, which would "
        "compute the expensive ESM2 embeddings twice. Model2's "
        f"{config.MODEL2_MIN_LEN}-{config.MODEL2_MAX_LEN}aa range applies to "
        "the whole request because Model2 cannot score outside it."
    ),
)
def predict_combined(body: PredictRequest, request: Request):
    started = time.perf_counter()
    client = _client(request)
    # Model2's 5-100aa range governs the combined call (documented above).
    clean, errors = validate_predict_sequences(
        body.sequences, enforce_length_range=True
    )
    if errors:
        _reject("/predict/combined", client, started, errors)
    app_state = request.app.state.app_state
    try:
        shared_emb = embed_esm2(clean)
        rows1, table1, thresholds = predict_model1(app_state.model1, clean, embeddings=shared_emb)
        rows2, table2, threshold2 = predict_model2(app_state.model2, clean, embeddings=shared_emb)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc)})
    merged = []
    for r1, r2 in zip(rows1, rows2):
        merged.append({"sequence": r1["sequence"], "model1": r1, "model2": r2})
    latency_ms = (time.perf_counter() - started) * 1000.0
    log_request(logger, endpoint="/predict/combined", client=client,
                n_sequences=len(clean), models=["model1", "model2"],
                latency_ms=latency_ms)
    return {
        "results": merged,
        "model1_table": table1,
        "model2_table": table2,
        "table_format": "csv",
        "model1_thresholds": thresholds,
        "model2_threshold": threshold2,
        "embedding_note": "ESM2 embeddings were computed once and shared by both models",
        "n_sequences": len(merged),
    }
