"""STANDING bit-equivalence guard for the fast object-native observation encoder.

`encode_observation` (used in collection) reads engine CardInstance objects directly and
re-derives the viewer's visibility filter + post-text-change fields fishrl-side, skipping the
UI-oriented `current_view`. That is the one risk this optimisation carries that a pure micro-opt
wouldn't: the lean path can silently DIVERGE from the view exactly in the states where those
fields are dynamic. So this test compares it byte-for-byte against the reference
`encode_observation_ref` (the dict/current_view path that is the behavioural contract).

Two deliberate properties (per the review guards):

1. STRATIFIED, not uniform. A green run must certify the HARD states were actually exercised,
   so we tally the four conditions where the lean logic is non-trivial and FAIL if any is unhit:
     * text-altered objects present  (Mind Bend / Crystal Spray / Vision Charm rewrote a card)
     * opponent hidden cards present (the known/hidden hand split)
     * known library slots present   (post scry / Brainstorm / Ponder reveals)
     * compound builder active        (builder_progress > 0)
   Uniform sampling would under-weight the rare altered-text states that are the entire reason
   the fields are dynamic. A synthetic text-change case backstops the text stratum so the guard
   is deterministic, not dependent on random play happening to fire a text-changer.

2. STANDING, not one-time. It runs in the DEFAULT suite (no slow gate) and is tied to the engine
   behaviour: if an engine re-vendor changes the visibility or text-change rules, the lean
   encoder diverges and THIS test fails. Re-run it on every re-vendor; do not delete it.
"""
import numpy as np

from fishrl.data.features import encode_god, encode_god_ref, encode_public, encode_public_ref
from fishrl.env.aec_env import FishAEC
from fishrl.forgetful_fish.state import text_variant
from fishrl.obs.encoder import encode_observation, encode_observation_ref

STRATA = ("text_altered", "opp_hidden", "known_library", "builder_active")


def _strata_for(g, viewer: str, prog: float) -> dict:
    opp = "p2" if viewer == "p1" else "p1"
    return {
        "text_altered": any(text_variant(o) for o in g.objects.values()),
        "opp_hidden": any(viewer not in (g.objects[iid].known_by or [])
                          for iid in g.players[opp].hand),
        "known_library": any(s.known_by.get(viewer) for s in g.library),
        "builder_active": prog > 0.0,
    }


def _assert_equal(g, viewer: str, prog: float, where: str) -> None:
    fast = encode_observation(g, viewer, prog)
    ref = encode_observation_ref(g, viewer, prog)
    if not np.array_equal(fast, ref):
        i = int(np.flatnonzero(fast != ref)[0])
        raise AssertionError(
            f"{where}: fast encoder diverges from current_view at index {i} "
            f"(fast={fast[i]} ref={ref[i]}, viewer={viewer}, prog={prog})")


def _play_and_compare(seeds, max_iter=2000) -> dict:
    """Random self-play across `seeds`; at every acting decision assert the fast and reference
    encoders agree for BOTH seats and tally which strata each comparison covered."""
    hits = {s: 0 for s in STRATA}
    comparisons = 0
    rng = np.random.default_rng(0)
    env = FishAEC(max_decisions=max_iter)
    for sd in seeds:
        env.reset(seed=sd)
        for agent in env.agent_iter(max_iter=max_iter * 6):
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
            obs = env.observe(agent)
            prog = env._builder.progress() if env._builder is not None else 0.0
            _assert_equal(env.g, agent, prog, f"seed={sd} acting")
            opp = "p2" if agent == "p1" else "p1"
            _assert_equal(env.g, opp, 0.0, f"seed={sd} non-acting")    # both seats covered
            for nm, fast_fn, ref_fn in (("encode_god", encode_god, encode_god_ref),
                                        ("encode_public", encode_public, encode_public_ref)):
                a, b = fast_fn(env.g), ref_fn(env.g)
                if not np.array_equal(a, b):
                    j = int(np.flatnonzero(a != b)[0])
                    raise AssertionError(f"seed={sd}: {nm} diverges at index {j} "
                                         f"(fast={a[j]} ref={b[j]})")
            for s, on in _strata_for(env.g, agent, prog).items():
                hits[s] += int(on)
            comparisons += 1
            legal = np.flatnonzero(obs["action_mask"])
            env.step(int(rng.choice(legal)))
    hits["_comparisons"] = comparisons
    return hits


def test_fast_encoder_matches_reference_stratified():
    hits = _play_and_compare(range(8))      # ~2000 comparisons, all strata covered, ~6s
    assert hits["_comparisons"] > 500, f"too few comparisons: {hits['_comparisons']}"
    # Every non-trivial stratum must be exercised; an empty one means the guard isn't actually
    # covering the case it claims to (uniform-sampling blind spot), so fail loudly.
    for s in ("opp_hidden", "known_library", "builder_active"):
        assert hits[s] > 0, f"stratum {s!r} never exercised (hits={hits})"


def test_fast_encoder_matches_reference_on_text_change():
    """Deterministic backstop for the text-altered stratum: rewrite a battlefield card the way a
    text-changer (Mind Bend: Island -> Swamp) does -- type_line/oracle_text rewritten AND the
    text_changes chain set, so text_variant(o) fires -- then assert the lean path still matches.
    This is the divergence the review flagged: reading the post-rewrite fields off the object the
    SAME way current_view does. Independent of whether random play happened to cast the spell."""
    env = FishAEC()
    env.reset(seed=3)
    target = None
    for iid in env.g.players["p1"].battlefield + env.g.players["p2"].battlefield:
        o = env.g.objects[iid]
        if "Island" in (o.type_line or "") or "Island" in (o.oracle_text or ""):
            target = o
            break
    if target is None:                          # no Island permanent yet; nothing to rewrite
        target = next(iter(env.g.objects.values()))
        target.type_line = "Basic Land — Island"
        target.oracle_text = "({T}: Add {U}.)"
    target.type_line = (target.type_line or "").replace("Island", "Swamp")
    target.oracle_text = (target.oracle_text or "").replace("Island", "Swamp")
    target.text_changes = [{"frm": "Island", "to": "Swamp"}]
    assert text_variant(target) == "swamp"      # the function the lean path must call
    for viewer in ("p1", "p2"):
        _assert_equal(env.g, viewer, 0.0, "synthetic text-change")


def test_fast_encoder_chain_without_string_change():
    """The static-template cache keys on (name, type_line, oracle_text, text_variant) --
    text_variant EXPLICITLY, because a text-change chain can exist on a card whose printed
    text never contained the changed word: the strings stay identical to the base card while
    the variant bit fires. Two objects that differ ONLY in that chain must not share a
    template. (This is the one way a string-only cache key would silently corrupt rows.)"""
    env = FishAEC()
    env.reset(seed=5)
    o = next(iter(env.g.objects.values()))
    base_tl, base_ot = o.type_line, o.oracle_text
    for viewer in ("p1", "p2"):
        _assert_equal(env.g, viewer, 0.0, "pre chain-only change")   # seeds the base template
    o.text_changes = [{"frm": "Island", "to": "Swamp"}]              # chain, strings untouched
    assert (o.type_line, o.oracle_text) == (base_tl, base_ot)
    assert text_variant(o) == "swamp"
    for viewer in ("p1", "p2"):
        _assert_equal(env.g, viewer, 0.0, "chain-only text change")
    o.text_changes = []                                              # and back: base again
    for viewer in ("p1", "p2"):
        _assert_equal(env.g, viewer, 0.0, "chain removed")


def test_template_cache_never_leaks_dynamic_state():
    """Same-identity cards with different dynamic state (tapped etc.) share one template;
    the dynamic writes must fully cover everything object-specific. Flip one card's tapped
    bit back and forth: the re-encoding must exactly reproduce the original vector (a
    template polluted by a previous encode would fail the round trip)."""
    env = FishAEC()
    env.reset(seed=7)
    bf = env.g.players["p1"].battlefield
    if not bf:                                   # ensure at least one battlefield object
        bf.append(next(iter(env.g.objects)))
    o = env.g.objects[bf[0]]
    before = encode_observation(env.g, "p1", 0.0)
    o.tapped = not o.tapped
    mid = encode_observation(env.g, "p1", 0.0)
    assert not np.array_equal(before, mid)       # the flip is visible
    o.tapped = not o.tapped
    after = encode_observation(env.g, "p1", 0.0)
    assert np.array_equal(before, after)         # perfect round trip: no cache pollution
    _assert_equal(env.g, "p1", 0.0, "post round-trip vs reference")
