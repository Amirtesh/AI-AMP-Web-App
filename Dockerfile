FROM python:3.11-slim

WORKDIR /app

# CPU-only torch first: Hugging Face Spaces runs on CPU hardware, and the
# default torch wheel ships CUDA (~2.5GB). Pre-installing the CPU build here
# satisfies `torch>=2.0.0` in requirements.txt so pip does not pull CUDA later.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# xgboost>=2.0.0 without its optional CUDA stack: xgboost 3.x declares
# nvidia-nccl-cu12 (~350MB) as a dependency that CPU inference never touches.
# Pre-installing dependency-free makes the requirements.txt line a no-op
# (numpy/scipy arrive with the rest of the file below).
RUN pip install --no-cache-dir --no-deps "xgboost>=2.0.0"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Bake the ESM2 checkpoint into the image at build time, not first request,
# so a cold-start after Spaces' free-tier sleep doesn't re-download ~150MB.
RUN python -c "import esm; esm.pretrained.esm2_t12_35M_UR50D()"


COPY backend/ ./backend/
COPY frontend/ ./frontend/


COPY Model1/model_artifacts/ ./Model1/model_artifacts/
COPY Model1/external_validation_summary_filtered.csv ./Model1/external_validation_summary_filtered.csv


COPY Model2/model2_esm2.joblib ./Model2/model2_esm2.joblib
COPY Model2/ablation_results_with_thresholds.csv ./Model2/ablation_results_with_thresholds.csv
COPY Model2/External-validation/ ./Model2/External-validation/
# NOTE: Model2/feature_columns.json is intentionally NOT copied — it does not
# exist in this repo. The backend derives the ESM2 column schema from the
# joblib artifact at startup (see README "Step 0 conclusion").


COPY Generative-Model/stage1_checkpoints/best_model.pt ./Generative-Model/stage1_checkpoints/best_model.pt
COPY Generative-Model/stage2_checkpoints/best_model.pt ./Generative-Model/stage2_checkpoints/best_model.pt


ENV PYTHONUNBUFFERED=1
EXPOSE 7860


CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
