# AI-AMP Model Suite — FastAPI Backend

Backend-only service (no frontend in this pass) over the four model components in `Web-App/`:
Model1 spectrum classifier, Model2 AMP gatekeeper, Stage 1 general-peptide generator, Stage 2 AMP-finetuned generator.
No RL / toxicity components are included here.

## Step 0 conclusion (why one service, one environment)

The three per-model `env.yml` files were diffed on python / torch / fair-esm / xgboost /
scikit-learn / joblib: all pin `python=3.11`, `numpy>=1.24,<2.0`, `torch>=2.0`, and where present
identical `xgboost>=2.0` / `scikit-learn>=1.3` ranges. Differences are purely additive, so this
ships as a **single FastAPI service with one unified environment** (`env.yml` in this directory).
Key inspection findings baked into the code:

- Model1's real inference path is `Model1/model_artifacts/*.json` + `config.json` (native XGBoost
  boosters); `Model1_final.pkl` is never referenced by `predict.py` and is ignored. Thresholds
  gp=0.50 / gn=0.52 / fungal=0.48 come from `config.json`. Only the gram-negative head uses ESM2;
  gram-positive/fungal use 34 biophysical features.
- Model1 enforces **no** training max length in its code (alphabet + non-empty only) — the API does
  not invent one. Model2 enforces 5–100aa plus an evidenced low-confidence flag under 15aa.
- `Model2/feature_columns.json` is missing on disk; columns are derived from the joblib's
  `n_features_in_` and asserted against the fixed ESM2-t12 width (480).
- `train_stage1.py` is missing under `Web-App/Generative-Model/`; the exact `SmallPeptideGPT`
  architecture + vocab were vendored from `Peptide-generation/train_stage1.py` into
  `backend/peptide_gpt.py` (checkpoint shapes re-verified). Generation hard cap: **83aa**.

## Setup

```bash
cd Web-App
conda env create -f env.yml
conda activate ai-amp-backend
```

## Run

```bash
cd Web-App
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Then open the web console: **http://localhost:8000/app** (Predict / Generate /
Validation tabs, served same-origin by the API — no CORS setup needed).
API docs: http://localhost:8000/docs.

Optional env vars: `DEVICE` (auto|cuda|cpu), `GITHUB_REPO_URL` (full-download link shown in
`GET /` and the OpenAPI docs), `PREDICT_PER_MINUTE_PER_IP` (default 60),
`GENERATE_PER_MINUTE_PER_IP` (default 10, stricter — generation is autoregressive),
`GENERATE_TIMEOUT_SEC` (default 180). All models load once at startup; the service refuses to
start if any artifact fails to load.

## Endpoints

| Method | Path | Input | Output |
|---|---|---|---|
| POST | `/predict/model1` | `{"sequences": [...]}` (max 100, 20 standard AA) | per-sequence gram+/gram−/fungal probs + 0/1 calls, `thresholds`, and a `table` CSV string mirroring `predictions.csv` |
| POST | `/predict/model2` | same contract + 5–100aa range | per-sequence AMP probability + AMP/non-AMP call, low-confidence note <15aa, `table` CSV, `threshold`, `validation_metrics` |
| POST | `/predict/combined` | same contract (5–100aa governs) | both results per sequence + both CSV tables; **computes ESM2 once and shares it — the frontend should use this instead of calling the two single-model endpoints back to back** |
| POST | `/generate/stage1` | `{"n_sequences": n, ...}` (max 100) | sequences + `amp_validated: false` and a warning: general-peptide samples, NOT equivalent to Stage 2 |
| POST | `/generate/stage2` | same shape | sequences + `amp_validated: true` (AMP-finetuned deliverable) |
| GET | `/stats` | — | precomputed metrics (CSVs as JSON) plus `images`/`files` URL lists — metrics and plots are separate sections |
| GET | `/stats/images/{name}` | — | static PNG (Model2 benchmark plots, stage1/2 loss curves + comparison) |
| GET | `/stats/files/{name}` | — | static CSV (Model2 ablation, Model1 external-validation summary) |
| GET | `/` | — | service info + `github_repository` full-download link |

Validation failures return 422 naming the exact sequence index, offending character(s), or limit;
rate limits return 429 with `Retry-After`; generation timeouts return 504. Every request is logged
as one JSON line with input size, model(s) invoked, latency, and any rejection reason.

## Layout

```
backend/main.py            app + lifespan startup (fail loudly) + GET /
backend/config.py          paths, limits, thresholds source of truth
backend/validation.py      sequence validation (422-ready error details)
backend/esm2.py            single shared embed_esm2(), batched, loaded once
backend/biophysical.py     Model1 34-feature computation (vendored)
backend/peptide_gpt.py     SmallPeptideGPT arch + vocab (vendored, verified)
backend/model_loader.py    startup loading for all four models
backend/inference.py       real Model1/Model2 prediction + CSV-table builders
backend/generation.py      top-p autoregressive sampling (83aa cap)
backend/rate_limit.py      per-IP limits (stricter for /generate/*)
backend/routers/{predict,generate,stats}.py
frontend/index.html        AI-AMP Studio console (served at GET /app)
frontend/styles.css        "abyssal laboratory" theme + animations
frontend/app.js            same-origin API client, canvas background
```
