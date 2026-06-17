"""Entity encoder: shapes, occupancy masking, and parameter savings.

(The encoder's ability to *fit* outcomes is exercised by the A/B harness
``python -m fishrl.eval.ab_encoder`` rather than a unit test, to keep the suite
free of optimizer-loop / self-play-collection runs.)"""
import torch

from fishrl.models.entity_encoder import EntityEncoder, layout_from_slots
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


def test_occupancy_mask_empty_zone_pools_to_zero():
    enc = EntityEncoder(*layout_from_slots(SLOTS), ACTOR_IN)
    with torch.no_grad():
        out = enc(torch.zeros(1, ACTOR_IN))       # all-zero obs => every zone empty
    card_part = out[0, : len(SLOTS) * 2 * enc.d]   # pooled card features (before globals)
    assert float(card_part.abs().max()) == 0.0
