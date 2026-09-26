#!/usr/bin/env python3
"""
AMP Model 1 - Prediction script (native XGBoost JSON + config, no pickle)

Usage:
  python predict.py --sequence PEPTIDESEQ -all
  python predict.py --fasta input.fasta -gp -fungal
  python predict.py --csv input.csv --column sequence -gn
  python predict.py --txt input.txt -all

Requires a model_artifacts/ directory (default) containing:
  config.json, gram_positive_model.json, gram_negative_model.json, fungal_model.json
"""

import argparse
import pandas as pd
import numpy as np
import json
import os
import sys

import xgboost as xgb
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from modlamp.descriptors import GlobalDescriptor

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ---------- Feature computation ----------

def compute_biophysical_features(seq):
    seq = seq.upper().strip()
    if len(seq) == 0 or not set(seq) <= STANDARD_AA:
        return None
    features = {}
    for aa in STANDARD_AA:
        features[f"frac_{aa}"] = seq.count(aa) / len(seq)
    try:
        pa = ProteinAnalysis(seq)
        features["gravy"] = pa.gravy()
        helix, turn, sheet = pa.secondary_structure_fraction()
        features["ss_helix_frac"] = helix
        features["ss_turn_frac"] = turn
        features["ss_sheet_frac"] = sheet
    except Exception:
        return None
    try:
        desc = GlobalDescriptor(seq)
        desc.calculate_all()
        for name, val in zip(desc.featurenames, desc.descriptor[0]):
            features[f"modlamp_{name}"] = val
    except Exception:
        return None
    return features


def get_esm2_embeddings(sequences, device="cuda"):
    import torch
    import esm
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    all_emb = []
    for i in range(0, len(sequences), 32):
        batch = [(str(j), s) for j, s in enumerate(sequences[i:i+32])]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            results = model(tokens, repr_layers=[12])
        reps = results["representations"][12]
        for k, (_, s) in enumerate(batch):
            all_emb.append(reps[k, 1:len(s)+1].mean(0).cpu().numpy())
    return np.vstack(all_emb)


# ---------- Model loading (native JSON, no pickle) ----------

def load_model1(artifacts_dir):
    config_path = os.path.join(artifacts_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"ERROR: config.json not found in {artifacts_dir}")
        sys.exit(1)
    with open(config_path) as f:
        config = json.load(f)

    for task in ["gram_positive", "gram_negative", "fungal"]:
        model_path = os.path.join(artifacts_dir, f"{task}_model.json")
        if not os.path.exists(model_path):
            print(f"ERROR: {model_path} not found")
            sys.exit(1)
        booster = xgb.Booster()
        booster.load_model(model_path)
        config[task]["booster"] = booster

    return config


# ---------- Input parsing ----------

def parse_input(args):
    if args.sequence:
        return [args.sequence.strip()]
    if args.fasta:
        seqs, current = [], []
        with open(args.fasta) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current:
                        seqs.append("".join(current)); current = []
                elif line:
                    current.append(line)
            if current:
                seqs.append("".join(current))
        return seqs
    if args.txt:
        with open(args.txt) as f:
            return [line.strip() for line in f if line.strip()]
    if args.csv:
        if not args.column:
            print("ERROR: --column must be specified when using --csv")
            sys.exit(1)
        df = pd.read_csv(args.csv)
        if args.column not in df.columns:
            print(f"ERROR: column '{args.column}' not found. Available: {df.columns.tolist()}")
            sys.exit(1)
        return df[args.column].dropna().astype(str).tolist()
    print("ERROR: must provide one of --sequence, --fasta, --csv, --txt")
    sys.exit(1)


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Model 1 - AMP Gram+/Gram-/Fungal predictor")
    ig = parser.add_mutually_exclusive_group(required=True)
    ig.add_argument("--sequence", type=str)
    ig.add_argument("--fasta", type=str)
    ig.add_argument("--csv", type=str)
    ig.add_argument("--txt", type=str)
    parser.add_argument("--column", type=str, help="Required with --csv")

    tg = parser.add_argument_group("tasks")
    tg.add_argument("-gp", "--gram_positive", action="store_true")
    tg.add_argument("-gn", "--gram_negative", action="store_true")
    tg.add_argument("-fungal", "--fungal", action="store_true")
    tg.add_argument("-all", "--all", action="store_true")

    parser.add_argument("--model_dir", type=str, default="model_artifacts",
                         help="Directory containing config.json and *_model.json files")
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    tasks = ["gram_positive", "gram_negative", "fungal"] if args.all else \
            [t for t, flag in [("gram_positive", args.gram_positive),
                                ("gram_negative", args.gram_negative),
                                ("fungal", args.fungal)] if flag]
    if not tasks:
        print("ERROR: specify -gp, -gn, -fungal, or -all")
        sys.exit(1)

    config = load_model1(args.model_dir)

    raw_sequences = parse_input(args)
    print(f"Loaded {len(raw_sequences)} input sequences")

    seqs, dropped = [], []
    for s in raw_sequences:
        s_clean = s.upper().strip()
        if len(s_clean) == 0 or not set(s_clean) <= STANDARD_AA:
            dropped.append(s)
        else:
            seqs.append(s_clean)
    print(f"Valid: {len(seqs)} | Dropped (non-standard/empty): {len(dropped)}")
    if dropped:
        print("  Examples dropped:", dropped[:5])
    if not seqs:
        print("No valid sequences. Exiting.")
        sys.exit(1)

    needs_bio = any(config[t]["feature_set_name"] == "biophysical" for t in tasks)
    needs_esm2 = any(config[t]["feature_set_name"] == "esm2" for t in tasks)

    bio_df = None
    if needs_bio:
        print("Computing biophysical features...")
        rows = []
        for s in seqs:
            feats = compute_biophysical_features(s)
            if feats is not None:
                feats["sequence"] = s
                rows.append(feats)
        bio_df = pd.DataFrame(rows)
        print(f"  Featurized: {len(bio_df)}/{len(seqs)}")

    esm2_df = None
    if needs_esm2:
        print("Computing ESM2 embeddings...")
        emb = get_esm2_embeddings(seqs, device=args.device)
        esm2_cols = [f"esm2_{i}" for i in range(emb.shape[1])]
        esm2_df = pd.DataFrame(emb, columns=esm2_cols)
        esm2_df["sequence"] = seqs

    if bio_df is not None and esm2_df is not None:
        base_df = pd.merge(bio_df, esm2_df, on="sequence")
    else:
        base_df = bio_df if bio_df is not None else esm2_df

    results = base_df[["sequence"]].copy()

    for task in tasks:
        info = config[task]
        feature_cols = info["features"]
        missing = [c for c in feature_cols if c not in base_df.columns]
        if missing:
            print(f"ERROR: missing features for '{task}': {missing[:5]}...")
            sys.exit(1)

        dmatrix = xgb.DMatrix(base_df[feature_cols])
        probs = info["booster"].predict(dmatrix)
        thresh = info["threshold"]

        results[f"{task}_prob"] = probs
        results[f"{task}_pred"] = (probs >= thresh).astype(int)
        print(f"{task}: threshold={thresh:.3f}, positives={results[f'{task}_pred'].sum()}/{len(results)}")

    results.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
