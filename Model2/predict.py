#!/usr/bin/env python3
"""
Model 2 (AMP/non-AMP gatekeeper) prediction script.

Selected model: ESM2-only (esm2_t12_35M_UR50D, layer 12, mean-pooled).
Chosen over the combined (biophysical+ESM2) model because a bootstrap CI on
the internal test set showed no statistically significant difference between
them (95% CI on MCC difference: [-0.0088, 0.0115], straddles zero) — ESM2-only
is simpler and has no biophysical-feature-pipeline dependency at inference
time, so it's preferred on a tie rather than arbitrarily.

USAGE:
    python3 predict.py --sequence GLFDIVKKVVGALGSL
    python3 predict.py --input peptides.txt
    python3 predict.py --input peptides.csv --column sequence
    python3 predict.py --input peptides.csv --column seq --output my_predictions.csv
"""

import argparse
import sys
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import torch
import esm

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
SHORT_LENGTH_FLAG_THRESHOLD = 15  # below this, flag low confidence per evidenced 31-65% sensitivity range


def load_sequences(args):
    """Returns a DataFrame with at least a 'sequence' column, preserving input order."""
    if args.sequence:
        return pd.DataFrame({"sequence": [args.sequence]})

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[FATAL] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if input_path.suffix.lower() == ".csv":
        if not args.column:
            print("[FATAL] --column is required when --input is a .csv file.", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(input_path)
        if args.column not in df.columns:
            print(f"[FATAL] Column '{args.column}' not found in {input_path}. "
                  f"Actual columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)
        out = df.copy()
        out["sequence"] = out[args.column]
        return out

    else:
        # plain text, one sequence per line
        lines = [l.strip() for l in input_path.read_text().splitlines()]
        lines = [l for l in lines if l]  # drop blank lines
        return pd.DataFrame({"sequence": lines})


def validate_sequences(df):
    """Adds 'valid' and 'invalid_reason' columns. Does NOT drop rows — invalid
    sequences are kept with a clear reason so the output CSV accounts for
    every input row, rather than silently vanishing."""
    df = df.copy()
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    df["length"] = df["sequence"].str.len()

    reasons = []
    for seq in df["sequence"]:
        if len(seq) == 0:
            reasons.append("empty sequence")
        elif not set(seq) <= STANDARD_AA:
            reasons.append("contains non-standard amino acid characters")
        elif len(seq) < 5:
            reasons.append("below 5aa — outside the model's training length range")
        elif len(seq) > 100:
            reasons.append("above 100aa — outside the model's training length range")
        else:
            reasons.append("")
    df["invalid_reason"] = reasons
    df["valid"] = df["invalid_reason"] == ""
    return df


def get_embeddings(sequences, device, batch_size=32):
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    all_emb = []
    for i in range(0, len(sequences), batch_size):
        batch = [(str(j), s) for j, s in enumerate(sequences[i:i + batch_size])]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            results = model(tokens, repr_layers=[12])
        reps = results["representations"][12]
        for k, (_, s) in enumerate(batch):
            all_emb.append(reps[k, 1:len(s) + 1].mean(0).cpu().numpy())
    return np.vstack(all_emb)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--sequence", type=str, help="A single peptide sequence.")
    group.add_argument("--input", type=str, help="Path to a .txt (one sequence per line) or .csv file.")
    ap.add_argument("--column", type=str, default=None,
                     help="Column name containing sequences. Required if --input is a .csv file.")
    ap.add_argument("--output", type=str, default="predictions.csv")
    args = ap.parse_args()

    # Model, feature schema, and threshold all live alongside this script.
    script_dir = Path(__file__).resolve().parent

    print("Loading sequences...")
    df = load_sequences(args)
    print(f"  {len(df)} sequence(s) loaded")

    df = validate_sequences(df)
    n_invalid = (~df["valid"]).sum()
    if n_invalid:
        print(f"  [warn] {n_invalid} invalid sequence(s) — will be marked in output, not silently dropped")

    # --- Load model, threshold, expected feature columns ---
    model_path = script_dir / "model2_esm2.joblib"
    if not model_path.exists():
        print(f"[FATAL] Model not found at {model_path}", file=sys.stderr)
        sys.exit(1)
    clf = joblib.load(model_path)

    with open(script_dir / "feature_columns.json") as f:
        cols = json.load(f)
    esm2_cols = cols["esm2_cols"]

    ablation = pd.read_csv(script_dir / "ablation_results_with_thresholds.csv")
    thresh_rows = ablation[ablation["feature_set"] == "esm2"]
    if len(thresh_rows) == 0:
        print("[FATAL] No 'esm2' row found in ablation_results_with_thresholds.csv", file=sys.stderr)
        sys.exit(1)
    threshold = float(thresh_rows["threshold"].iloc[0])
    print(f"  Using threshold: {threshold}")

    # --- Embed only the valid sequences ---
    valid_df = df[df["valid"]].copy()
    if len(valid_df) == 0:
        print("[FATAL] No valid sequences to predict on.", file=sys.stderr)
        df.to_csv(args.output, index=False)
        print(f"Wrote {args.output} (all rows invalid, no predictions made)")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Computing ESM2 embeddings on {device} ({len(valid_df)} sequences)...")
    emb_matrix = get_embeddings(valid_df["sequence"].tolist(), device)
    assert emb_matrix.shape[0] == len(valid_df), "Embedding count mismatch — stop and debug"

    X = pd.DataFrame(emb_matrix, columns=esm2_cols, index=valid_df.index)

    print("Predicting...")
    probs = clf.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    df["probability"] = np.nan
    df["prediction"] = ""
    df.loc[valid_df.index, "probability"] = probs
    df.loc[valid_df.index, "prediction"] = np.where(preds == 1, "AMP", "non-AMP")

    # --- Confidence flag: evidenced short-sequence weakness, not decorative ---
    def confidence_note(row):
        if not row["valid"]:
            return ""
        if row["length"] < SHORT_LENGTH_FLAG_THRESHOLD:
            return (f"LOW CONFIDENCE: sequence is under {SHORT_LENGTH_FLAG_THRESHOLD}aa. "
                     f"External validation measured sensitivity as low as 31% in this length range "
                     f"(likely an ESM2 mean-pooling effect, not a training data gap).")
        return ""

    df["confidence_note"] = df.apply(confidence_note, axis=1)

    df.to_csv(args.output, index=False)
    print(f"\nWrote {args.output} ({len(df)} rows: {len(valid_df)} predicted, {n_invalid} invalid)")
    if n_invalid:
        print(f"  invalid rows are included with an 'invalid_reason' and no prediction, not dropped silently")


if __name__ == "__main__":
    main()
