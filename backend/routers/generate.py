"""POST /generate/stage1 and /generate/stage2."""

import asyncio
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .. import config
from ..generation import generate_sequences
from ..logging_setup import log_request, setup_logging
from ..rate_limit import check_generate_rate

logger = setup_logging()
router = APIRouter(prefix="/generate", tags=["generate"])


class GenerateRequest(BaseModel):
    n_sequences: int = Field(..., ge=1, le=100, description="Number of sequences to generate (max 100)")
    min_length: int | None = Field(
        default=None, ge=1, le=config.GENERATIVE_MAX_LEN,
        description="Minimum peptide length (EOS blocked until reached)",
    )
    max_length: int | None = Field(
        default=None, ge=1, le=config.GENERATIVE_MAX_LEN,
        description=f"Maximum peptide length (hard cap {config.GENERATIVE_MAX_LEN}, the trained positional ceiling)",
    )
    temperature: float = Field(default=config.GENERATE_DEFAULT_TEMPERATURE, gt=0.0, le=2.0)
    top_p: float = Field(default=config.GENERATE_DEFAULT_TOP_P, gt=0.0, le=1.0)


STAGE1_LABEL = {
    "model": "stage1_general_peptide",
    "amp_validated": False,
    "warning": (
        "These are general-peptide-corpus (PeptideAtlas-pretrained) samples, "
        "NOT AMP-validated. Do not present Stage 1 output as equivalent to "
        "Stage 2 AMP-finetuned output."
    ),
}
STAGE2_LABEL = {
    "model": "stage2_amp_finetuned",
    "amp_validated": True,
    "note": "AMP-fine-tuned model — the validated deliverable for AMP design.",
}


def _validate_gen(body: GenerateRequest) -> str | None:
    if (
        body.min_length is not None
        and body.max_length is not None
        and body.min_length > body.max_length
    ):
        return (
            f"min_length ({body.min_length}) cannot exceed max_length "
            f"({body.max_length})"
        )
    for name, val in (("min_length", body.min_length), ("max_length", body.max_length)):
        if val is not None and val > config.GENERATIVE_MAX_LEN:
            return (
                f"{name} ({val}) exceeds the model's trained positional limit "
                f"({config.GENERATIVE_MAX_LEN}) — generation would exceed the "
                "positional embeddings, so it is rejected, not truncated"
            )
    return None


async def _run(stage: str, label: dict, body: GenerateRequest, request: Request):
    started = time.perf_counter()
    client = request.client.host if request.client else "unknown"
    problem = _validate_gen(body)
    if problem:
        latency_ms = (time.perf_counter() - started) * 1000.0
        log_request(logger, endpoint=f"/generate/{stage}", client=client,
                    n_sequences=body.n_sequences, models=[f"generative_{stage}"],
                    latency_ms=latency_ms, rejection=problem)
        raise HTTPException(status_code=422, detail={"message": problem})
    gen = request.app.state.app_state.generative
    model = gen.stage1 if stage == "stage1" else gen.stage2
    try:
        seqs, meta = await asyncio.wait_for(
            asyncio.to_thread(
                generate_sequences, model, gen.device, body.n_sequences,
                body.min_length, body.max_length, body.temperature, body.top_p,
            ),
            timeout=config.GENERATE_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        latency_ms = (time.perf_counter() - started) * 1000.0
        reason = (
            f"generation exceeded the {config.GENERATE_TIMEOUT_SEC:g}s request "
            "timeout; try fewer sequences or a smaller max_length"
        )
        log_request(logger, endpoint=f"/generate/{stage}", client=client,
                    n_sequences=body.n_sequences, models=[f"generative_{stage}"],
                    latency_ms=latency_ms, rejection=reason)
        raise HTTPException(
            status_code=504,
            detail={"message": reason},
        )
    latency_ms = (time.perf_counter() - started) * 1000.0
    log_request(logger, endpoint=f"/generate/{stage}", client=client,
                n_sequences=len(seqs), models=[f"generative_{stage}"],
                latency_ms=latency_ms)
    return {
        **label,
        "sequences": seqs,
        "n_sequences": len(seqs),
        "max_length_cap": config.GENERATIVE_MAX_LEN,
        "parameters": {
            "min_length": body.min_length,
            "max_length": body.max_length,
            "temperature": body.temperature,
            "top_p": body.top_p,
        },
    }


@router.post(
    "/stage1",
    dependencies=[Depends(check_generate_rate)],
    summary="Stage 1 general-peptide generation (NOT AMP-validated)",
    description=(
        "Samples from the PeptideAtlas-pretrained general-peptide model. "
        "The response carries amp_validated=false plus a warning the frontend "
        "must surface: Stage 1 output is NOT equivalent to Stage 2. "
        f"Max generation length is capped at {config.GENERATIVE_MAX_LEN}."
    ),
)
async def generate_stage1(body: GenerateRequest, request: Request):
    return await _run("stage1", STAGE1_LABEL, body, request)


@router.post(
    "/stage2",
    dependencies=[Depends(check_generate_rate)],
    summary="Stage 2 AMP-finetuned generation (validated deliverable)",
    description=(
        "Samples from the AMP-fine-tuned model (amp_validated=true). "
        f"Max generation length is capped at {config.GENERATIVE_MAX_LEN}, "
        "the positional-embedding ceiling inherited from Stage 1."
    ),
)
async def generate_stage2(body: GenerateRequest, request: Request):
    return await _run("stage2", STAGE2_LABEL, body, request)
