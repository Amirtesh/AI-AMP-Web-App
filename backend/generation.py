"""Autoregressive peptide generation for Stage 1 / Stage 2.

Port of generate_batch() from Generative-Model/generate_base.py (identical
in generate_finetuned.py): EOS blocked until min_len, hard stop at max_len
(capped by the 83aa positional ceiling), temperature + nucleus (top-p)
sampling. The per-request path generates fresh samples without the
bulk-collection novelty bookkeeping (training_seq.csv resume loop), which
does not apply to single API calls.
"""

from . import config


def generate_batch(model, device, batch_size, min_len, max_len,
                   temperature=1.0, top_p=0.9):
    import torch
    import torch.nn.functional as F

    from .peptide_gpt import BOS_IDX, EOS_IDX, IDX2TOKEN, PAD_IDX

    hard_cap = max_len if max_len is not None else config.GENERATIVE_MAX_LEN
    hard_cap = min(hard_cap, config.GENERATIVE_MAX_LEN)
    ids = torch.full((batch_size, 1), BOS_IDX, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    with torch.no_grad():
        for _ in range(hard_cap):
            logits = model(ids)[:, -1, :] / temperature
            cur_len = ids.shape[1] - 1
            if min_len is not None and cur_len < min_len:
                logits[:, EOS_IDX] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff_mask = cumsum > top_p
            cutoff_mask[:, 0] = False  # always keep top-1 token available
            sorted_probs = sorted_probs.masked_fill(cutoff_mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_ids = sorted_idx.gather(
                -1, torch.multinomial(sorted_probs, 1)
            ).squeeze(-1)
            next_ids = torch.where(
                finished, torch.full_like(next_ids, PAD_IDX), next_ids
            )
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


def generate_sequences(model, device, n_sequences: int, min_len, max_len,
                       temperature: float, top_p: float) -> tuple:
    """Collect n_sequences non-empty, run-unique samples. Returns (seqs, meta)."""
    collected: list = []
    seen = set()
    n_raw = 0
    batch_size = min(config.GENERATE_SAMPLING_BATCH, max(8, n_sequences))
    guard = 0
    while len(collected) < n_sequences:
        guard += 1
        if guard > 200:
            raise RuntimeError(
                f"Generation stalled: only {len(collected)}/{n_sequences} "
                "unique non-empty sequences after 200 batches"
            )
        batch = generate_batch(model, device, batch_size, min_len, max_len,
                               temperature, top_p)
        n_raw += len(batch)
        for seq in batch:
            if not seq or seq in seen:
                continue
            seen.add(seq)
            collected.append(seq)
            if len(collected) >= n_sequences:
                break
    meta = {"n_raw_batches": n_raw, "max_len_cap": config.GENERATIVE_MAX_LEN}
    return collected, meta
