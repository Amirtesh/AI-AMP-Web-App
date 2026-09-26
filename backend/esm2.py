"""Single shared ESM2 embedding function.

Dedup of the identical computation in Model1/predict.py:get_esm2_embeddings
and Model2/predict.py:get_embeddings (same checkpoint esm2_t12_35M_UR50D,
same repr layer 12, same residue mean-pooling, same batch size 32).

torch / fair-esm are imported lazily so that importing this module (and the
FastAPI app) never requires GPU libraries — they are only needed inside the
running service, loaded once at startup via init_embedder().
"""

import numpy as np

from . import config

_embedder = None


class ESM2Embedder:
    """Batched ESM2 embedder: one forward pass per batch, never per-sequence."""

    def __init__(self, device: str = "auto"):
        import torch
        import esm

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        self.model = model.to(device).eval()
        self.batch_converter = alphabet.get_batch_converter()
        self._torch = torch

    def embed(self, sequences: list) -> np.ndarray:
        torch = self._torch
        all_emb = []
        batch_size = config.ESM2_BATCH_SIZE
        for i in range(0, len(sequences), batch_size):
            batch = [(str(j), s) for j, s in enumerate(sequences[i : i + batch_size])]
            _, _, tokens = self.batch_converter(batch)
            tokens = tokens.to(self.device)
            with torch.no_grad():
                results = self.model(tokens, repr_layers=[config.ESM2_REPR_LAYER])
            reps = results["representations"][config.ESM2_REPR_LAYER]
            for k, (_, s) in enumerate(batch):
                # Mean-pool over residue positions only (skip BOS at 0).
                all_emb.append(reps[k, 1 : len(s) + 1].mean(0).cpu().numpy())
        return np.vstack(all_emb)


def init_embedder(device: str = "auto") -> ESM2Embedder:
    """Load the ESM2 model once at startup. Raises loudly on failure."""
    global _embedder
    try:
        _embedder = ESM2Embedder(device=device)
    except Exception as exc:
        raise RuntimeError(f"Failed to load ESM2 model: {exc}") from exc
    return _embedder


def embed_esm2(sequences: list) -> np.ndarray:
    """Shared entry point used by Model1 and Model2 predictors.

    /predict/combined calls this exactly once and feeds the same matrix to
    both heads — never call ESM2 twice per request.
    """
    if _embedder is None:
        raise RuntimeError("ESM2 embedder is not initialized (startup failed?)")
    if not sequences:
        raise ValueError("embed_esm2 received an empty sequence list")
    return _embedder.embed(sequences)
