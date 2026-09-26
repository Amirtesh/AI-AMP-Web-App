"""SmallPeptideGPT — exact architecture from Peptide-generation/train_stage1.py.

Copied verbatim (vocab, CausalSelfAttentionBlock, SmallPeptideGPT) because
Web-App/Generative-Model ships only the generate_*.py scripts and the
*.pt state_dicts, not the training module they import. The checkpoint key
names (token_emb, pos_emb, blocks.0-5, ln_f, head) and tensor shapes
(token_emb 23x256, pos_emb 85x256) were verified against the actual
best_model.pt files before vendoring — do not modify without re-verifying.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401  (kept for parity with source)

# --------------------------------------------------------------------------
# Vocab (fixed, not a CLI arg - changing this would break checkpoint compat)
# --------------------------------------------------------------------------
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>"]
VOCAB = SPECIAL_TOKENS + AMINO_ACIDS
TOKEN2IDX = {tok: i for i, tok in enumerate(VOCAB)}
IDX2TOKEN = {i: tok for i, tok in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
PAD_IDX = TOKEN2IDX["<PAD>"]
BOS_IDX = TOKEN2IDX["<BOS>"]
EOS_IDX = TOKEN2IDX["<EOS>"]


def encode_sequence(seq, max_len):
    ids = [BOS_IDX] + [TOKEN2IDX[c] for c in seq] + [EOS_IDX]
    pad_len = (max_len + 2) - len(ids)
    ids = ids + [PAD_IDX] * pad_len
    return ids


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class CausalSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, causal_mask):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class SmallPeptideGPT(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=256, n_heads=8, n_layers=6, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            CausalSelfAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), diagonal=1)
        for block in self.blocks:
            x = block(x, causal_mask)
        x = self.ln_f(x)
        return self.head(x)
