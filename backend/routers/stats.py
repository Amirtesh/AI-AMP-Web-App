"""GET /stats — precomputed validation artifacts, served as-is.

Nothing is recomputed here (no UMAP/clustering re-runs): CSVs are parsed to
JSON and PNGs are served as static files. Metrics and images are separated
so the frontend can show result tables and plots in distinct areas.
"""

import csv

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .. import config

router = APIRouter(prefix="/stats", tags=["stats"])


def _read_csv(path) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# Whitelisted static files: name -> (path, description).
IMAGES = {
    "model2_benchmark_metrics_bar": (
        config.MODEL2_EXTVAL_DIR / "benchmark_metrics_bar.png",
        "Model2 external-validation benchmark metrics bar plot",
    ),
    "model2_class_imbalance_bar": (
        config.MODEL2_EXTVAL_DIR / "class_imbalance_bar.png",
        "Model2 external-validation class imbalance bar plot",
    ),
    "stage1_vs_stage2_comparison": (
        config.GENERATIVE_DIR / "stage1_vs_stage2_comparison.png",
        "Stage 1 vs Stage 2 comparison plot (generative models)",
    ),
    "stage1_loss_curve": (
        config.GENERATIVE_DIR / "stage1_loss_curve.png",
        "Stage 1 training loss curve",
    ),
    "stage1_lr_schedule": (
        config.GENERATIVE_DIR / "stage1_lr_schedule.png",
        "Stage 1 learning-rate schedule",
    ),
    "stage1_step_loss": (
        config.GENERATIVE_DIR / "stage1_step_loss.png",
        "Stage 1 per-step loss",
    ),
    "stage2_loss_curve": (
        config.GENERATIVE_DIR / "stage2_loss_curve.png",
        "Stage 2 training loss curve",
    ),
    "stage2_lr_schedule": (
        config.GENERATIVE_DIR / "stage2_lr_schedule.png",
        "Stage 2 learning-rate schedule",
    ),
    "stage2_step_loss": (
        config.GENERATIVE_DIR / "stage2_step_loss.png",
        "Stage 2 per-step loss",
    ),
}

FILES = {
    "model2_ablation_results_with_thresholds": (
        config.MODEL2_ABLATION_CSV,
        "Model2 ablation results with thresholds (CSV)",
    ),
    "model1_external_validation_summary": (
        config.MODEL1_EXTERNAL_VAL_CSV,
        "Model1 external validation summary, filtered (CSV)",
    ),
}


@router.get(
    "",
    summary="Precomputed validation results (metrics + links to static plots)",
    description=(
        "Static artifacts from completed validation runs, served as-is. "
        "Metrics tables and plot images are returned in separate sections "
        "(\"metrics\" vs \"images\") so the frontend can display model metrics "
        "and images in distinct areas. Nothing is recomputed on request."
    ),
)
def get_stats():
    m2 = _read_csv(config.MODEL2_ABLATION_CSV)
    m1 = _read_csv(config.MODEL1_EXTERNAL_VAL_CSV)
    return {
        "metrics": {
            "model1": {
                "thresholds": dict(_model1_thresholds()),
                "external_validation_summary": m1,
            },
            "model2": {
                "ablation_results_with_thresholds": m2,
            },
            "generative": {
                "max_generation_length": config.GENERATIVE_MAX_LEN,
                "note": "Stage 2 inherits Stage 1's 83aa positional-embedding ceiling.",
            },
        },
        "images": [
            {"name": name, "url": f"/stats/images/{name}", "description": desc}
            for name, (_, desc) in IMAGES.items()
        ],
        "files": [
            {"name": name, "url": f"/stats/files/{name}", "description": desc}
            for name, (_, desc) in FILES.items()
        ],
        "note": "All artifacts are pre-generated results served as static files; nothing is recomputed.",
    }


def _model1_thresholds() -> dict:
    import json

    cfg_path = config.MODEL1_ARTIFACTS_DIR / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    return {t: cfg[t]["threshold"] for t in ("gram_positive", "gram_negative", "fungal")}


@router.get(
    "/images/{name}",
    summary="Serve a pre-generated validation plot (PNG)",
    response_class=FileResponse,
)
def get_image(name: str):
    if name not in IMAGES:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Unknown image '{name}'",
                "available": sorted(IMAGES),
            },
        )
    path, _ = IMAGES[name]
    if not path.exists():
        raise HTTPException(
            status_code=404, detail={"message": f"Artifact file missing: {path.name}"}
        )
    return FileResponse(str(path), media_type="image/png")


@router.get(
    "/files/{name}",
    summary="Serve a pre-generated validation CSV",
    response_class=FileResponse,
)
def get_file(name: str):
    if name not in FILES:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Unknown file '{name}'",
                "available": sorted(FILES),
            },
        )
    path, _ = FILES[name]
    if not path.exists():
        raise HTTPException(
            status_code=404, detail={"message": f"Artifact file missing: {path.name}"}
        )
    return FileResponse(str(path), media_type="text/csv")
