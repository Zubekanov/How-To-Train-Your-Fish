"""STANDING equivalence guard for the C zone-encoder (`fishrl._speed._fastenc`).

When the extension is built, `encoder._fill_zone` IS the C path -- so the whole
encoder-equivalence suite already exercises it against the reference encoders.
This file adds the DIRECT comparison: the C loop vs the pure-Python twin
(`_fill_zone_py`) on the same states, row-for-row, which localises a divergence
to the C code instead of surfacing it as a distant encoder mismatch. Skipped
entirely on checkouts that haven't built the extension (Linux deploy, ODROID,
fresh clones) -- those run the Python twin and are covered by the existing suite.
"""
import numpy as np
import pytest

from fishrl.env.aec_env import FishAEC
from fishrl.obs import encoder as enc

_fastenc = pytest.importorskip("fishrl._speed._fastenc")


def _zone_lists(g, viewer: str):
    """The exact zone object-lists encode_observation builds (hidden slots as None)."""
    opp = "p2" if viewer == "p1" else "p1"
    obj = g.objects
    return [
        ([obj[iid] for iid in g.players[viewer].hand], enc.SLOTS["own_hand"]),
        ([obj[iid] if viewer in (obj[iid].known_by or []) else None
          for iid in g.players[opp].hand], enc.SLOTS["opp_hand"]),
        ([obj[iid] for iid in g.players[viewer].battlefield], enc.SLOTS["own_bf"]),
        ([obj[iid] for iid in g.players[opp].battlefield], enc.SLOTS["opp_bf"]),
        ([obj[iid] for iid in g.graveyard], enc.SLOTS["graveyard"]),
        ([obj[iid] for iid in g.exile], enc.SLOTS["exile"]),
        ([obj.get(s.source_instance_id) for s in g.stack], enc.SLOTS["stack"]),
        ([obj[s.instance_id] for s in g.library if s.known_by.get(viewer)],
         enc.SLOTS["library"]),
    ]


def test_c_fill_zone_matches_python():
    rng = np.random.default_rng(0)
    env = FishAEC(max_decisions=1500)
    compared = 0
    for sd in range(4):
        env.reset(seed=sd)
        for agent in env.agent_iter(max_iter=6000):
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
            obs = env.observe(agent)
            for viewer in ("p1", "p2"):
                for objs, n in _zone_lists(env.g, viewer):
                    c_rows = np.zeros((n, enc.CARD_F), dtype=np.float32)
                    p_rows = np.zeros((n, enc.CARD_F), dtype=np.float32)
                    nb = _fastenc.fill_zone(c_rows, 0, objs, n, viewer)
                    assert nb == enc._fill_zone_py(p_rows, 0, objs, n, viewer)
                    if not np.array_equal(c_rows, p_rows):
                        r, c = np.argwhere(c_rows != p_rows)[0]
                        raise AssertionError(
                            f"seed={sd} viewer={viewer}: C row {r} col {c} "
                            f"= {c_rows[r, c]} vs python {p_rows[r, c]}")
                    compared += 1
            legal = np.flatnonzero(obs["action_mask"])
            env.step(int(rng.choice(legal)))
    assert compared > 2000, f"too few zone comparisons: {compared}"


def test_c_fill_zone_shares_cache_and_rejects_bad_buffers():
    env = FishAEC()
    env.reset(seed=11)
    # library slots always exist (a fresh hand may not be dealt yet at reset)
    objs = [env.g.objects[s.instance_id] for s in env.g.library[:6]]
    assert objs, "expected a populated library at reset"
    rows = np.zeros((len(objs), enc.CARD_F), dtype=np.float32)
    before = len(enc._ROW_CACHE)
    _fastenc.fill_zone(rows, 0, objs, len(objs), "p1")
    # misses went through the Python _build_row, so they landed in the SHARED cache
    assert len(enc._ROW_CACHE) >= before
    key = next(iter(enc._ROW_CACHE))
    np.testing.assert_array_equal(rows[0][: enc.CARD_F].shape, (enc.CARD_F,))
    assert isinstance(key, tuple) and len(key) == 11
    # wrong dtype must be an error, never silent mis-indexing
    with pytest.raises(TypeError):
        _fastenc.fill_zone(np.zeros((4, enc.CARD_F)), 0, objs, 4, "p1")
    # base/n outside the buffer must be an error
    with pytest.raises(ValueError):
        _fastenc.fill_zone(np.zeros((2, enc.CARD_F), dtype=np.float32), 0, objs, 3, "p1")
