"""Shared card-embedding ("entity") encoders.

The flat observation is a deterministic concatenation of per-card rows (one block
per zone, each ``n_slots × CARD_F``) followed by a globals tail. A flat MLP over
that vector must relearn each card's meaning independently in every (zone, slot) —
"Memory Lapse in hand" and "Memory Lapse on top of library" are unrelated input
dimensions. These encoders instead reshape the SAME flat vector back into per-card
rows and apply one **shared** card encoder to every row, so a card's
representation is learned once and reused everywhere; a zone embedding and a
within-zone positional embedding (giving explicit library depth) are added.

Two pooling strategies share that front-end:

* :class:`EntityEncoder` — masked mean + max per zone. Cheap, permutation-
  invariant, but lossy: two distinct boards with the same per-feature mean/extremum
  collapse to the same summary, and cards never "see" each other.
* :class:`AttentionEntityEncoder` — a multi-head self-attention block lets every
  occupied card attend to every other card *across zones* (a hand card can condition
  on what is on the battlefield / in the graveyard), then a learned per-zone
  attention pool replaces the masked mean with a content-weighted sum. This is the
  hypothesised fix for flat-beats-entity on the A/B (eval/ab_encoder): the relational
  signal mean/max pooling throws away.

Both are pure model-side reinterpretations — no change to the observation, the env,
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


class _EntityBase(nn.Module):
    """Shared front-end: reshape flat obs -> tokens, add zone/pos embeddings, read
    occupancy. Subclasses implement :meth:`_pool` to turn ``(tokens, occ)`` per zone
    into the pooled feature block."""

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

        zone_ids, pos_ids = [], []
        for name, n in zip(zone_names, self.zone_slots):
            zone_ids += [ZONE_VOCAB[name]] * n
            pos_ids += list(range(n))
        self.register_buffer("zone_ids", torch.tensor(zone_ids, dtype=torch.long))
        self.register_buffer("pos_ids", torch.tensor(pos_ids, dtype=torch.long))
        # Row -> output-zone index (0..len(zone_slots)-1, in layout order). Lets the
        # per-zone pool run as ONE scatter-reduce over all R rows instead of an 8-iteration
        # Python loop of ~6 tiny ops each -- the loop was ~70% of the batch-1 encoder cost
        # (pure op-dispatch overhead on the collection hot path).
        seg_ids = [i for i, n in enumerate(self.zone_slots) for _ in range(n)]
        self.register_buffer("seg_ids", torch.tensor(seg_ids, dtype=torch.long))

    def _tokens(self, x: torch.Tensor):
        """flat obs -> (tokens (B,R,d), occupancy (B,R), globals (B,G))."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B = x.shape[0]
        ent = x[:, :self.R * CARD_F].reshape(B, self.R, CARD_F)
        glob = x[:, self.R * CARD_F:]
        occ = ((ent[..., _UNK_IDX] + ent[..., _KNOWN_IDX]) > 0).float()    # (B, R)
        h = self.card(ent) + self.zone_emb(self.zone_ids) + self.pos_emb(self.pos_ids)
        return h, occ, glob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, occ, glob = self._tokens(x)
        h = self._contextualize(h, occ)
        outs, start = [], 0
        for n in self.zone_slots:
            outs += self._pool(h[:, start:start + n, :], occ[:, start:start + n])
            start += n
        return torch.cat(outs + [glob], dim=-1)

    def _contextualize(self, h: torch.Tensor, occ: torch.Tensor) -> torch.Tensor:
        return h                                          # no cross-entity mixing by default

    def _pool(self, seg: torch.Tensor, m: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError


class EntityEncoder(_EntityBase):
    """Masked mean + max per zone (2*d per zone), globals appended."""

    @property
    def enc_dim(self) -> int:
        return len(self.zone_slots) * 2 * self.d + self.globals_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Vectorized equivalent of _EntityBase.forward's per-zone loop: one segment
        # mean + one segment max over ALL rows, keyed on seg_ids, then interleaved
        # [mean_z, max_z] per zone in layout order (matching the loop's cat order).
        # Numerically identical to the loop up to float sum-order (<1e-6).
        h, occ, glob = self._tokens(x)
        h = self._contextualize(h, occ)                    # identity for this encoder
        B, Z, d = h.shape[0], len(self.zone_slots), self.d
        occ1 = occ.unsqueeze(-1)                            # (B, R, 1)
        idx_d = self.seg_ids.view(1, -1, 1).expand(B, -1, d)
        zsum = h.new_zeros(B, Z, d).scatter_add(1, idx_d, h * occ1)
        zcnt = occ1.new_zeros(B, Z, 1).scatter_add(
            1, self.seg_ids.view(1, -1, 1).expand(B, -1, 1), occ1)
        zmean = zsum / zcnt.clamp(min=1.0)                  # empty zone -> 0/1 = 0
        hmask = h.masked_fill(occ1 == 0, float("-inf"))
        zmax = h.new_full((B, Z, d), float("-inf")).scatter_reduce(
            1, idx_d, hmask, reduce="amax", include_self=True)
        zmax = torch.where(torch.isinf(zmax), torch.zeros_like(zmax), zmax)  # empty -> 0
        zc = torch.cat([zmean, zmax], dim=-1).reshape(B, Z * 2 * d)
        return torch.cat([zc, glob], dim=-1)

    def _pool(self, seg: torch.Tensor, m: torch.Tensor) -> list[torch.Tensor]:
        m = m.unsqueeze(-1)                                            # (B, n, 1)
        mean = (seg * m).sum(1) / m.sum(1).clamp(min=1.0)             # masked mean
        mx = seg.masked_fill(m == 0, float("-inf")).max(1).values     # masked max
        mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)   # empty zone -> 0
        return [mean, mx]


class AttentionEntityEncoder(_EntityBase):
    """Cross-zone self-attention contextualisation, then a learned attention pool
    (+ masked max) per zone (2*d per zone), globals appended.

    enc_dim matches :class:`EntityEncoder`, so the A/B compares pooling strategy at
    (near-)matched width rather than confounding it with the head MLP's input size.
    """

    def __init__(self, *args, n_heads: int = 4, n_layers: int = 1, **kw):
        super().__init__(*args, **kw)
        self.attn = nn.ModuleList(
            nn.MultiheadAttention(self.d, n_heads, batch_first=True) for _ in range(n_layers))
        self.attn_ln = nn.ModuleList(nn.LayerNorm(self.d) for _ in range(n_layers))
        self.pool_score = nn.Linear(self.d, 1)        # per-token logit for the attention pool

    @property
    def enc_dim(self) -> int:
        return len(self.zone_slots) * 2 * self.d + self.globals_dim

    def _contextualize(self, h: torch.Tensor, occ: torch.Tensor) -> torch.Tensor:
        # key_padding_mask: True positions are IGNORED as keys. A row with zero
        # occupied tokens would mask every key and make softmax produce NaN, so let
        # such rows attend to slot 0 (its pooled output is discarded by occupancy).
        kpm = occ == 0
        kpm = kpm.clone()
        kpm[kpm.all(dim=1), 0] = False
        for attn, ln in zip(self.attn, self.attn_ln):
            a, _ = attn(h, h, h, key_padding_mask=kpm, need_weights=False)
            h = ln(h + a)
        return h

    def _pool(self, seg: torch.Tensor, m: torch.Tensor) -> list[torch.Tensor]:
        s = self.pool_score(seg).squeeze(-1)                          # (B, n)
        s = s.masked_fill(m == 0, float("-inf"))
        w = torch.nan_to_num(Fn.softmax(s, dim=1), nan=0.0)          # empty zone -> 0 weights
        pooled = (seg * w.unsqueeze(-1)).sum(1)                       # content-weighted sum
        mx = seg.masked_fill(m.unsqueeze(-1) == 0, float("-inf")).max(1).values
        mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)   # empty zone -> 0
        return [pooled, mx]


def layout_from_slots(slots: dict) -> tuple[list[str], list[int]]:
    """(zone_names, zone_slots) from a SLOTS-style ordered dict."""
    return list(slots.keys()), list(slots.values())


def build_entity_encoder(encoder: str, zone_names: list[str], zone_slots: list[int],
                         total_in: int, d: int = 64) -> _EntityBase | None:
    """Encoder factory: ``"entity"`` / ``"attention"`` -> module, ``"flat"`` -> None.
    `d` is the per-card embedding width (Config.card_dim); it sets enc_dim, so the
    head MLP that consumes this encoder adapts automatically."""
    if encoder == "entity":
        return EntityEncoder(zone_names, zone_slots, total_in, d=d)
    if encoder == "attention":
        return AttentionEntityEncoder(zone_names, zone_slots, total_in, d=d)
    return None
