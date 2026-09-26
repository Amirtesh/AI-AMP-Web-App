"""Startup model loading. All heavy imports are function-local so that
`import backend.main` never requires torch/xgboost/sklearn — those are only
needed inside the running service. Every loader raises RuntimeError with an
explicit message; the lifespan handler refuses to start on any failure —
no silent fallback, no partial startup.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .esm2 import init_embedder


@dataclass
class Model1State:
    tasks: dict = field(default_factory=dict)  # task -> {booster, features, threshold, feature_set_name}
    thresholds: dict = field(default_factory=dict)


@dataclass
class Model2State:
    clf: object = None
    threshold: float = 0.5
    esm2_cols: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


@dataclass
class GenerativeState:
    stage1: object = None
    stage2: object = None
    device: str = "cpu"
    max_len: int = config.GENERATIVE_MAX_LEN


@dataclass
class AppState:
    model1: Model1State
    model2: Model2State
    generative: GenerativeState
    esm2_device: str = "cpu"


def resolve_device() -> str:
    import torch

    want = config.DEVICE.lower()
    if want == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu" if want in ("cpu", "cuda") else ("cuda" if torch.cuda.is_available() else "cpu")


def load_model1() -> Model1State:
    """Real inference path: model_artifacts/*.json + config.json (native
    XGBoost boosters). Model1_final.pkl is legacy and intentionally ignored —
    Model1/predict.py never references it (see Step 0 report)."""
    import xgboost as xgb

    artifacts_dir = config.MODEL1_ARTIFACTS_DIR
    config_path = artifacts_dir / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"Model1 config.json not found in {artifacts_dir}")
    with open(config_path) as f:
        raw = json.load(f)
    state = Model1State()
    for task in ("gram_positive", "gram_negative", "fungal"):
        if task not in raw:
            raise RuntimeError(f"Model1 config.json is missing task '{task}'")
        model_path = artifacts_dir / f"{task}_model.json"
        if not model_path.exists():
            raise RuntimeError(f"Model1 booster not found: {model_path}")
        try:
            booster = xgb.Booster()
            booster.load_model(str(model_path))
        except Exception as exc:
            raise RuntimeError(f"Failed to load Model1 booster {model_path}: {exc}") from exc
        info = raw[task]
        state.tasks[task] = {
            "booster": booster,
            "features": info["features"],
            "threshold": float(info["threshold"]),
            "feature_set_name": info["feature_set_name"],
        }
        state.thresholds[task] = float(info["threshold"])
    return state


def load_model2() -> Model2State:
    """model2_esm2.joblib (XGBClassifier) + threshold from
    ablation_results_with_thresholds.csv (feature_set == 'esm2' row).

    NOTE: Model2/predict.py expects a feature_columns.json that is absent
    from Model2/ on disk. The ESM2-t12 embedding width is fixed at 480
    (independently confirmed by Model1's config.json esm2_0..esm2_479), so
    the columns are derived from clf.n_features_in_ and asserted — no
    invented schema, and a wrong artifact fails loudly here, not per-request.
    """
    import joblib
    import pandas as pd

    if not config.MODEL2_JOBLIB.exists():
        raise RuntimeError(f"Model2 artifact not found: {config.MODEL2_JOBLIB}")
    try:
        clf = joblib.load(str(config.MODEL2_JOBLIB))
    except Exception as exc:
        raise RuntimeError(f"Failed to unpickle Model2 joblib: {exc}") from exc

    if not config.MODEL2_ABLATION_CSV.exists():
        raise RuntimeError(f"Model2 ablation CSV not found: {config.MODEL2_ABLATION_CSV}")
    ablation = pd.read_csv(str(config.MODEL2_ABLATION_CSV))
    rows = ablation[ablation["feature_set"] == "esm2"]
    if len(rows) == 0:
        raise RuntimeError("No 'esm2' row in Model2 ablation_results_with_thresholds.csv")
    row = rows.iloc[0]
    threshold = float(row["threshold"])
    metrics = {k: float(row[k]) for k in ("test_AUROC", "test_AUPRC", "test_MCC", "test_F1")}

    if config.MODEL2_FEATURE_COLUMNS_JSON.exists():
        with open(config.MODEL2_FEATURE_COLUMNS_JSON) as f:
            esm2_cols = json.load(f)["esm2_cols"]
    else:
        n_feat = int(getattr(clf, "n_features_in_", config.ESM2_EMB_DIM))
        if n_feat != config.ESM2_EMB_DIM:
            raise RuntimeError(
                f"Model2 expects {n_feat} features but ESM2-t12 produces "
                f"{config.ESM2_EMB_DIM}; refusing to guess a column mapping"
            )
        esm2_cols = [f"esm2_{i}" for i in range(n_feat)]
    return Model2State(clf=clf, threshold=threshold, esm2_cols=esm2_cols, metrics=metrics)


def load_generative(device: str) -> GenerativeState:
    """Stage 1 + Stage 2 SmallPeptideGPT state_dicts (same arch, same 83aa
    positional ceiling — Stage 2 inherits it unchanged)."""
    import torch

    from .peptide_gpt import VOCAB_SIZE, SmallPeptideGPT

    def _load(checkpoint: Path):
        if not checkpoint.exists():
            raise RuntimeError(f"Generative checkpoint not found: {checkpoint}")
        model = SmallPeptideGPT(VOCAB_SIZE, config.GENERATIVE_MAX_LEN + 2,
                                d_model=256, n_heads=8, n_layers=6)
        try:
            state = torch.load(str(checkpoint), map_location=device)
        except Exception as exc:
            raise RuntimeError(f"Failed to torch.load {checkpoint}: {exc}") from exc
        # Checkpoints are plain state_dicts (see Step 0 report).
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        try:
            model.load_state_dict(state)
        except Exception as exc:
            raise RuntimeError(
                f"Checkpoint {checkpoint} does not match SmallPeptideGPT "
                f"(vocab={VOCAB_SIZE}, max_len={config.GENERATIVE_MAX_LEN + 2}): {exc}"
            ) from exc
        model.to(device).eval()
        return model

    stage1 = _load(config.STAGE1_CHECKPOINT)
    stage2 = _load(config.STAGE2_CHECKPOINT)
    return GenerativeState(stage1=stage1, stage2=stage2, device=device,
                           max_len=config.GENERATIVE_MAX_LEN)


def build_state() -> AppState:
    """Load everything once. Any failure raises — the app refuses to start."""
    device = resolve_device()
    init_embedder(device=device)  # ESM2 first: both predictors need it
    model1 = load_model1()
    model2 = load_model2()
    generative = load_generative(device)
    return AppState(model1=model1, model2=model2, generative=generative,
                    esm2_device=device)
