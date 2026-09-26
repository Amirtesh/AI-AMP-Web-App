"""Central configuration for the AI-AMP FastAPI backend.

All paths are anchored at the Web-App/ directory (parent of this package),
so the service runs regardless of the current working directory.
"""

import os
from pathlib import Path

# Web-App/ directory (this file lives at Web-App/backend/config.py).
WEBAPP_DIR = Path(__file__).resolve().parent.parent

MODEL1_ARTIFACTS_DIR = Path(
    os.environ.get("MODEL1_ARTIFACTS_DIR", WEBAPP_DIR / "Model1" / "model_artifacts")
)
MODEL2_JOBLIB = Path(
    os.environ.get("MODEL2_JOBLIB", WEBAPP_DIR / "Model2" / "model2_esm2.joblib")
)
MODEL2_ABLATION_CSV = Path(
    os.environ.get(
        "MODEL2_ABLATION_CSV",
        WEBAPP_DIR / "Model2" / "ablation_results_with_thresholds.csv",
    )
)
MODEL2_FEATURE_COLUMNS_JSON = Path(
    os.environ.get(
        "MODEL2_FEATURE_COLUMNS_JSON",
        WEBAPP_DIR / "Model2" / "feature_columns.json",
    )
)
MODEL1_EXTERNAL_VAL_CSV = WEBAPP_DIR / "Model1" / "external_validation_summary_filtered.csv"
MODEL2_EXTVAL_DIR = WEBAPP_DIR / "Model2" / "External-validation"
GENERATIVE_DIR = WEBAPP_DIR / "Generative-Model"
STAGE1_CHECKPOINT = Path(
    os.environ.get(
        "STAGE1_CHECKPOINT",
        GENERATIVE_DIR / "stage1_checkpoints" / "best_model.pt",
    )
)
STAGE2_CHECKPOINT = Path(
    os.environ.get(
        "STAGE2_CHECKPOINT",
        GENERATIVE_DIR / "stage2_checkpoints" / "best_model.pt",
    )
)

# Full-download link for the project. No URL was found in the repo, so this
# is env-configurable; set GITHUB_REPO_URL to the real repository address.
# It is surfaced in GET / and in the OpenAPI docs description.
GITHUB_REPO_URL = os.environ.get(
    "GITHUB_REPO_URL", "https://github.com/<your-org>/AI-AMP-Design"
)

# --- Request contracts (see README + Step 0 report) ---
MAX_SEQUENCES = 100
STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")

# Model2 validated length range (Model2/predict.py validate_sequences).
# Model1's predict.py enforces NO max length (empty + alphabet only), so
# Model1/model1-side of /combined applies alphabet validation only.
MODEL2_MIN_LEN = 5
MODEL2_MAX_LEN = 100
# Model2 short-sequence low-confidence flag (evidenced 31-65% sensitivity).
MODEL2_SHORT_FLAG_THRESHOLD = 15

# Generative positional-embedding ceiling (generate_base.py STAGE1_MAX_LEN).
GENERATIVE_MAX_LEN = 83

# Shared ESM2 batching (matches predict.py chunk size of 32).
ESM2_BATCH_SIZE = 32
ESM2_MODEL_NAME = "esm2_t12_35M_UR50D"
ESM2_REPR_LAYER = 12
# esm2_t12_35M embedding width; asserted against the joblib artifact at startup.
ESM2_EMB_DIM = 480

# --- Non-functional knobs ---
PREDICT_PER_MINUTE_PER_IP = int(os.environ.get("PREDICT_PER_MINUTE_PER_IP", "60"))
GENERATE_PER_MINUTE_PER_IP = int(os.environ.get("GENERATE_PER_MINUTE_PER_IP", "10"))
# Generation is autoregressive: bound it, never hang the worker.
GENERATE_TIMEOUT_SEC = float(os.environ.get("GENERATE_TIMEOUT_SEC", "180"))
GENERATE_DEFAULT_TEMPERATURE = 1.0
GENERATE_DEFAULT_TOP_P = 0.9
GENERATE_SAMPLING_BATCH = 64

DEVICE = os.environ.get("DEVICE", "auto")  # auto | cuda | cpu
