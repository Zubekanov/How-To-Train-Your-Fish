"""The checkpoint must record EVERY architecture knob the model shapes depend on, and
`config_from_checkpoint` must rebuild those shapes exactly -- otherwise a resized model
(bigger actor head, wider card embedding, entity encoder) fails to reload in the trainer
resume path, the eval panels, or the parallel collector. These guard that contract and
its backward compatibility with pre-resize checkpoints.
"""
from __future__ import annotations

import torch

from fishrl.train.config import Config
from fishrl.train.train_loop import (build_models, config_from_checkpoint, _encoders,
                                      _load_model_state, _model_state)


def _saved_config(cfg: Config) -> dict:
    """The architecture dict the trainer writes into the checkpoint (train_loop._payload)."""
    return {"seed": cfg.seed, "encoders": _encoders(cfg), "use_belief": cfg.use_belief,
            "critic_hidden": list(cfg.critic_hidden), "hidden": list(cfg.hidden),
            "actor_hidden": (list(cfg.actor_hidden) if cfg.actor_hidden is not None else None),
            "card_dim": cfg.card_dim}


def test_bigger_entity_model_roundtrips_through_the_saved_config():
    # A resized run: entity actor+critic, a deeper actor head, a wider card embedding.
    cfg = Config(device="cpu", encoder="flat", actor_encoder="entity", critic_encoder="entity",
                 actor_hidden=(768, 768, 384), card_dim=128)
    m = build_models(cfg)
    # rebuilding from ONLY the saved config dict must produce load-compatible shapes
    cfg2 = config_from_checkpoint(_saved_config(cfg), device="cpu")
    assert cfg2.card_dim == 128 and cfg2.head_hidden("actor") == (768, 768, 384)
    m2 = build_models(cfg2)
    _load_model_state(m2, _model_state(m))            # raises on any shape mismatch
    # the actor really is the bigger entity shape, not the default
    assert tuple(m2.actor.enc.card.name.weight.shape) == (128, 20)
    assert m2.actor.enc.enc_dim == m.actor.enc.enc_dim


def test_actor_hidden_overrides_only_the_actor():
    cfg = Config(actor_hidden=(768, 768, 384), card_dim=128)
    assert cfg.head_hidden("actor") == (768, 768, 384)
    assert cfg.head_hidden("guesser") == (256, 256)     # guesser stays on the default (hot path)
    assert cfg.head_hidden("critic") == (512, 512, 256)
    # None actor_hidden falls back to the shared hidden
    assert Config().head_hidden("actor") == (256, 256)


def test_pre_resize_checkpoint_reconstructs_the_old_flat_architecture():
    # An old checkpoint has no hidden/actor_hidden/card_dim keys at all.
    old = {"seed": 0,
           "encoders": {"actor": "flat", "critic": "entity", "guesser": "flat", "public": "flat"},
           "use_belief": True, "critic_hidden": [512, 512, 256]}
    cfg = config_from_checkpoint(old, device="cpu")
    assert cfg.card_dim == 64 and cfg.actor_hidden is None and cfg.head_hidden("actor") == (256, 256)
    m = build_models(cfg)
    assert m.actor.enc is None                          # flat actor (no entity encoder)
    assert tuple(m.actor.net[0].weight.shape)[1] == 6517  # ACTOR_IN, the flat input width


def test_overrides_win_over_saved_runtime_knobs_but_not_architecture():
    cfg = Config(actor_hidden=(512, 512), card_dim=128)
    saved = _saved_config(cfg)
    rebuilt = config_from_checkpoint(saved, device="cpu", pool_frac=0.9, iters=7)
    assert rebuilt.pool_frac == 0.9 and rebuilt.iters == 7   # runtime knobs applied
    assert rebuilt.card_dim == 128 and rebuilt.head_hidden("actor") == (512, 512)  # arch preserved
