#!/usr/bin/env python3
"""
Generate novel peptide sequences from the Stage 2 (AMP-fine-tuned) model.
Same novelty enforcement and interface as generate_base.py.
"""

import argparse
import os
import time

import pandas as pd
import torch
import torch.nn.functional as F

from train_stage1 import (
    SmallPeptideGPT, VOCAB_SIZE, IDX2TOKEN, PAD_IDX, BOS_IDX, EOS_IDX
)

CHECKPOINT_PATH = "stage2_checkpoints/best_model.pt"   # <-- only real difference
STAGE1_MAX_LEN = 83  # positional embedding ceiling - inherited from Stage 1, unchanged
OUTPUT_FILE = "generated_finetuned_model.csv"           # <-- and this


def load_model(device):
    model = SmallPeptideGPT(VOCAB_SIZE, STAGE1_MAX_LEN + 2, d_model=256, n_heads=8, n_layers=6)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_batch(model, device, batch_size, min_len, max_len, temperature=1.0, top_p=0.9):
    hard_cap = max_len if max_len is not None else STAGE1_MAX_LEN
    ids = torch.full((batch_size, 1), BOS_IDX, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(hard_cap):
        logits = model(ids)[:, -1, :] / temperature
        cur_len = ids.shape[1] - 1
        if min_len is not None and cur_len < min_len:
            logits[:, EOS_IDX] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        cutoff_mask = cumsum > top_p
        cutoff_mask[:, 0] = False
        sorted_probs = sorted_probs.masked_fill(cutoff_mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        next_ids = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1)).squeeze(-1)
        next_ids = torch.where(finished, torch.full_like(next_ids, PAD_IDX), next_ids)
        finished = finished | (next_ids == EOS_IDX)
        ids = torch.cat([ids, next_ids.unsqueeze(1)], dim=1)
        if finished.all():
            break

    sequences = []
    for row in ids.tolist():
        chars = []
        for tok in row[1:]:
            if tok in (EOS_IDX, PAD_IDX):
                break
            chars.append(IDX2TOKEN[tok])
        sequences.append("".join(chars))
    return sequences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseq", type=int, required=True)
    ap.add_argument("--min", type=int, default=None)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--training_seq", default="training_seq.csv")
    ap.add_argument("--output", default=OUTPUT_FILE)
    args = ap.parse_args()

    if args.min is not None and args.max is not None and args.min > args.max:
        raise ValueError(f"--min ({args.min}) cannot exceed --max ({args.max})")
    if args.max is not None and args.max > STAGE1_MAX_LEN:
        raise ValueError(f"--max ({args.max}) exceeds the model's positional limit ({STAGE1_MAX_LEN})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(args.training_seq):
        raise FileNotFoundError(f"{args.training_seq} not found - build it first (see merge command).")
    train_df = pd.read_csv(args.training_seq)
    training_set = set(train_df["sequence"].astype(str).str.strip().str.upper())
    print(f"Loaded {len(training_set)} training sequences for novelty check")

    model = load_model(device)
    print(f"Loaded model from {CHECKPOINT_PATH}")

    collected, seen_this_run = [], set()
    if os.path.exists(args.output):
        existing = pd.read_csv(args.output)
        collected = existing["sequence"].tolist()
        seen_this_run = set(collected)
        print(f"Resuming: {len(collected)} already collected in {args.output}")

    n_generated, n_duplicate = 0, 0
    start = time.time()

    while len(collected) < args.nseq:
        batch = generate_batch(model, device, args.batch_size, args.min, args.max,
                                args.temperature, args.top_p)
        n_generated += len(batch)
        for seq in batch:
            if not seq:
                continue
            if seq in training_set or seq in seen_this_run:
                n_duplicate += 1
                continue
            seen_this_run.add(seq)
            collected.append(seq)
            if len(collected) >= args.nseq:
                break

        elapsed = time.time() - start
        print(f"\r  collected {len(collected)}/{args.nseq} | raw {n_generated} | "
              f"dup {n_duplicate} | {elapsed/60:.1f}min", end="", flush=True)

        if len(collected) % (args.batch_size * 10) < args.batch_size:
            pd.DataFrame({"sequence": collected}).to_csv(args.output, index=False)

    pd.DataFrame({"sequence": collected}).to_csv(args.output, index=False)
    print(f"\nDone. {len(collected)} novel sequences -> {args.output}")
    print(f"Novelty rate: {(1 - n_duplicate/max(n_generated,1))*100:.1f}% of raw output was novel")


if __name__ == "__main__":
    main()
