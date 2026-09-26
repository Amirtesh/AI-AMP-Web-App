"""Prediction logic shared by the /predict/* endpoints.

Every function here calls the real loaded models (native XGBoost boosters
for Model1, XGBClassifier for Model2) on real ESM2 embeddings. No stubs.
Heavy imports (xgboost, pandas) are function-local.
"""

import io

from . import config
from .biophysical import biophysical_matrix
from .esm2 import embed_esm2

MODEL1_TASKS = ("gram_positive", "gram_negative", "fungal")


def _to_csv(rows: list, columns: list) -> str:
    import pandas as pd

    buf = io.StringIO()
    pd.DataFrame(rows, columns=columns).to_csv(buf, index=False)
    return buf.getvalue()


def predict_model1(state, sequences: list, embeddings=None) -> tuple:
    """Returns (rows, table_csv, thresholds).

    embeddings: optional precomputed ESM2 matrix (from the shared embed_esm2
    call) — /predict/combined passes it so ESM2 runs exactly once.
    """
    import numpy as np
    import pandas as pd
    import xgboost as xgb

    tasks = state.tasks
    needs_esm2 = any(
        tasks[t]["feature_set_name"] == "esm2" for t in MODEL1_TASKS
    )

    bio_df, failed = biophysical_matrix(
        sequences, tasks["gram_positive"]["features"]
    )
    if failed:
        idx = failed[0]
        raise ValueError(
            f"sequence[{idx}] passed alphabet validation but its biophysical "
            "features could not be computed (Bio/modlamp failure)"
        )

    emb_matrix = None
    if needs_esm2:
        emb_matrix = (
            np.asarray(embeddings)
            if embeddings is not None
            else embed_esm2(sequences)
        )
        if emb_matrix.shape[0] != len(sequences):
            raise RuntimeError(
                "ESM2 embedding count mismatch — refusing to predict"
            )

    rows = []
    for i, seq in enumerate(sequences):
        row = {"sequence": seq}
        for task in MODEL1_TASKS:
            info = tasks[task]
            if info["feature_set_name"] == "esm2":
                cols = info["features"]
                X = pd.DataFrame(
                    emb_matrix[i : i + 1], columns=cols[: emb_matrix.shape[1]]
                )
                # Guard against a feature-list/embedding width mismatch.
                if X.shape[1] != len(cols):
                    raise RuntimeError(
                        f"Model1 '{task}' expects {len(cols)} ESM2 features "
                        f"but the embedding has {X.shape[1]}"
                    )
                X = X[cols]
            else:
                X = bio_df.iloc[i : i + 1][info["features"]]
            prob = float(info["booster"].predict(xgb.DMatrix(X))[0])
            pred = int(prob >= info["threshold"])
            row[f"{task}_prob"] = prob
            row[f"{task}_pred"] = pred
        rows.append(row)

    columns = ["sequence"] + [
        c for t in MODEL1_TASKS for c in (f"{t}_prob", f"{t}_pred")
    ]
    table = _to_csv(rows, columns)
    thresholds = {t: tasks[t]["threshold"] for t in MODEL1_TASKS}
    return rows, table, thresholds


def predict_model2(state, sequences: list, embeddings=None) -> tuple:
    """Returns (rows, table_csv, threshold).

    Row shape mirrors Model2/predict.py's output CSV (sequence, length,
    probability, prediction, confidence_note) for the valid inputs the API
    accepts; invalid inputs are rejected with 422 before this runs.
    """
    import numpy as np
    import pandas as pd

    emb = (
        np.asarray(embeddings) if embeddings is not None else embed_esm2(sequences)
    )
    if emb.shape[0] != len(sequences):
        raise RuntimeError("ESM2 embedding count mismatch — refusing to predict")
    if emb.shape[1] != len(state.esm2_cols):
        raise RuntimeError(
            f"Model2 expects {len(state.esm2_cols)} ESM2 features but the "
            f"embedding has {emb.shape[1]}"
        )
    X = pd.DataFrame(emb, columns=state.esm2_cols)
    probs = state.clf.predict_proba(X)[:, 1]

    rows = []
    for seq, prob in zip(sequences, probs):
        prob = float(prob)
        pred = int(prob >= state.threshold)
        note = ""
        if len(seq) < config.MODEL2_SHORT_FLAG_THRESHOLD:
            note = (
                f"LOW CONFIDENCE: sequence is under "
                f"{config.MODEL2_SHORT_FLAG_THRESHOLD}aa. External validation "
                "measured sensitivity as low as 31% in this length range "
                "(likely an ESM2 mean-pooling effect, not a training data gap)."
            )
        rows.append(
            {
                "sequence": seq,
                "length": len(seq),
                "probability": prob,
                "prediction": "AMP" if pred == 1 else "non-AMP",
                "confidence_note": note,
            }
        )
    table = _to_csv(
        rows, ["sequence", "length", "probability", "prediction", "confidence_note"]
    )
    return rows, table, state.threshold
