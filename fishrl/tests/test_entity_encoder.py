"""Entity encoder: shapes, occupancy masking, and parameter savings.

(The encoder's ability to *fit* outcomes is exercised by the A/B harness
``python -m fishrl.eval.ab_encoder`` rather than a unit test, to keep the suite
free of optimizer-loop / self-play-collection runs.)"""
import pytest
import torch

from fishrl.models.entity_encoder import (
    AttentionEntityEncoder, EntityEncoder, build_entity_encoder, layout_from_slots)
from fishrl.models.policy import ACTOR_IN, MaskedActor
from fishrl.obs.encoder import SLOTS
from fishrl.spaces import action_space as A


def test_entity_actor_shapes_and_param_savings():
    flat, ent = MaskedActor(encoder="flat"), MaskedActor(encoder="entity")
    x = torch.zeros(3, ACTOR_IN)
    assert flat(x).shape == (3, A.N) and ent(x).shape == (3, A.N)
    pf = sum(p.numel() for p in flat.parameters())
    pe = sum(p.numel() for p in ent.parameters())
    assert pe < pf, f"entity ({pe}) should have fewer params than flat ({pf})"


@pytest.mark.parametrize("encoder", ["entity", "attention"])
def test_occupancy_mask_empty_zone_pools_to_zero(encoder):
    enc = build_entity_encoder(encoder, *layout_from_slots(SLOTS), ACTOR_IN)
    with torch.no_grad():
        out = enc(torch.zeros(1, ACTOR_IN))       # all-zero obs => every zone empty
    card_part = out[0, : len(SLOTS) * 2 * enc.d]   # pooled card features (before globals)
    assert float(card_part.abs().max()) == 0.0    # no NaN/Inf leaking from the empty softmax


def test_attention_actor_shapes_and_matched_enc_dim():
    attn = MaskedActor(encoder="attention")
    x = torch.zeros(3, ACTOR_IN)
    out = attn(x)
    assert out.shape == (3, A.N)
    assert torch.isfinite(out).all()
    # enc_dim matches the mean/max entity encoder, so the A/B isolates pooling strategy.
    ent_enc = EntityEncoder(*layout_from_slots(SLOTS), ACTOR_IN)
    assert attn.enc.enc_dim == ent_enc.enc_dim


def test_attention_mixes_across_entities():
    """Self-attention makes a zone's pooled output depend on OTHER zones' cards —
    a property masked mean/max pooling cannot have."""
    enc = AttentionEntityEncoder(*layout_from_slots(SLOTS), ACTOR_IN)
    enc.eval()
    torch.manual_seed(0)
    x = torch.zeros(1, ACTOR_IN)
    from fishrl.obs.encoder import CARD_F
    from fishrl.obs import vocab as V
    # Occupy own_hand slot 0 and opp_bf slot 0 (mark them "known").
    own_hand_row0 = 0
    opp_bf_row0 = (SLOTS["own_hand"] + SLOTS["opp_hand"] + SLOTS["own_bf"])
    for r in (own_hand_row0, opp_bf_row0):
        x[0, r * CARD_F + V.N_NAMES] = 1.0        # name one-hot stand-in / unknown bit -> occupied
    with torch.no_grad():
        base = enc(x)
        # Change a card in opp_bf; own_hand's pooled block should shift via attention.
        x2 = x.clone()
        x2[0, opp_bf_row0 * CARD_F + 0] = 1.0     # set a name feature on the opp_bf card
        moved = enc(x2)
    own_hand_block = slice(0, 2 * enc.d)          # own_hand is the first zone in SLOTS
    assert float((moved[0, own_hand_block] - base[0, own_hand_block]).abs().max()) > 1e-6
