"""Shared card-embedding ("entity") encoder.

The flat observation is a deterministic concatenation of per-card rows (one block
per zone, each ``n_slots × CARD_F``) followed by a globals tail. A flat MLP over
that vector must relearn each card's meaning independently in every (zone, slot) —
"Memory Lapse in hand" and "Memory Lapse on top of library" are unrelated input
dimensions. This encoder instead reshapes the SAME flat vector back into per-card
rows and applies one **shared** card encoder to every row, so a card's
representation is learned once and reused everywhere; a zone embedding and a
within-zone positional embedding (giving explicit library depth) are added, rows
are masked-pooled per zone, and the globals tail is concatenated.

It is a pure model-side reinterpretation — no change to the observation, the env,
or the buffers. Padding rows are all-zero; occupancy is read from the existing
unknown/known bits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from fishrl.obs.encoder import CARD_F
from fishrl.obs import vocab as V

_UNK_IDX = V.N_NAMES            # "unknown/hidden" bit position in a card row
_KNOWN_IDX = CARD_F - 1         # "known" bit position
_MAX_SLOTS = 64                 # largest zone (god-state library); for positional embedding

# Union of every zone name across the perspective / god / public layouts.
ZONE_VOCAB = {
    name: i for i, name in enumerate([
        "own_hand", "opp_hand", "own_bf", "opp_bf", "graveyard", "exile", "stack",
        "library", "p1_hand", "p2_hand", "p1_bf", "p2_bf",
    ])
}


class CardEncoder(nn.Module):
    """Per-card row -> d. Shared across all rows/zones (and optionally across nets)."""

    def __init__(self, d: int = 64):
        super().__init__()
        self.name = nn.Linear(V.N_NAMES, d, bias=False)          # embedding lookup over the one-hot
        self.feat = nn.Sequential(
            nn.Linear(CARD_F - V.N_NAMES, d), nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, d))
        self.ln = nn.LayerNorm(d)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        h = self.name(rows[..., :V.N_NAMES]) + self.feat(rows[..., V.N_NAMES:])
        return Fn.relu(self.ln(h))


class EntityEncoder(nn.Module):
    """Reshape a flat observation into card entities, encode/pool per zone, append globals."""

    def __init__(self, zone_names: list[str], zone_slots: list[int], total_in: int,
                 d: int = 64, card_encoder: CardEncoder | None = None):
        super().__init__()
        self.zone_slots = list(zone_slots)
        self.R = sum(self.zone_slots)
        self.globals_dim = total_in - self.R * CARD_F
        assert self.globals_dim >= 0, f"layout exceeds input ({self.R}*{CARD_F} > {total_in})"
        self.d = d
        self.card = card_encoder or CardEncoder(d)
        self.zone_emb = nn.Embedding(len(ZONE_VOCAB), d)
        self.pos_emb = nn.Embedding(_MAX_SLOTS, d)
        self.enc_dim = len(self.zone_slots) * 2 * d + self.globals_dim

        zone_ids, pos_ids = [], []
        for name, n in zip(zone_names, self.zone_slots):
            zone_ids += [ZONE_VOCAB[name]] * n
            pos_ids += list(range(n))
        self.register_buffer("zone_ids", torch.tensor(zone_ids, dtype=torch.long))
        self.register_buffer("pos_ids", torch.tensor(pos_ids, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B = x.shape[0]
        ent = x[:, :self.R * CARD_F].reshape(B, self.R, CARD_F)
        glob = x[:, self.R * CARD_F:]
        occ = ((ent[..., _UNK_IDX] + ent[..., _KNOWN_IDX]) > 0).float()    # (B, R)
        h = self.card(ent) + self.zone_emb(self.zone_ids) + self.pos_emb(self.pos_ids)

        outs = []
        start = 0
        for n in self.zone_slots:
            seg = h[:, start:start + n, :]                                  # (B, n, d)
            m = occ[:, start:start + n].unsqueeze(-1)                       # (B, n, 1)
            mean = (seg * m).sum(1) / m.sum(1).clamp(min=1.0)               # masked mean
            mx = seg.masked_fill(m == 0, float("-inf")).max(1).values       # masked max
            mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)     # empty zone -> 0
            outs += [mean, mx]
            start += n
        return torch.cat(outs + [glob], dim=-1)


def layout_from_slots(slots: dict) -> tuple[list[str], list[int]]:
    """(zone_names, zone_slots) from a SLOTS-style ordered dict."""
    return list(slots.keys()), list(slots.values())
