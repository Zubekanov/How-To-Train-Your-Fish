"""The relay must refuse to move a checkpoint between checkouts that encode the game differently.

The relay moves latest.pt in BOTH directions, so a feature-dim mismatch is destructive either
way: pulling imports an unloadable checkpoint over this machine's run, and handing back breaks
the remote's service the moment it resumes. The deckout clock (OBS_DIM 6494 -> 6497) is exactly
such a change, and the old preflight -- which only checked that the remote could import
fishrl.train.ownership -- would have waved it straight through.
"""
from fishrl.relay import feature_refusal, local_dims


def test_matching_checkouts_relay_fine():
    dims = (6497, 8920, 6456, 285)
    assert feature_refusal(dims, dims, "odroid-lan", "fishrl-selfplay") is None


def test_an_observation_change_is_refused_and_says_which_dim():
    local = (6497, 8920, 6456, 285)          # this checkout has the deckout clock
    remote = (6494, 8917, 6453, 285)         # the ODROID does not
    msg = feature_refusal(local, remote, "odroid-lan", "fishrl-selfplay")
    assert msg is not None
    assert "FEATURE MISMATCH" in msg
    assert "OBS_DIM" in msg and "6497" in msg and "6494" in msg
    assert "odroid-lan" in msg and "fishrl-selfplay" in msg
    assert "migrate" in msg                   # tells you the checkpoint needs migrating
    assert "action_space.N" not in msg        # only the dims that ACTUALLY differ are listed


def test_an_action_space_change_is_refused_too():
    # e.g. the BF 20 -> 34 cap: same observation rows, different action space
    msg = feature_refusal((6497, 8920, 6456, 285), (6497, 8920, 6456, 243), "h", "u")
    assert msg is not None and "action_space.N" in msg


def test_local_dims_are_ints_and_match_the_live_encoders():
    from fishrl.data.features import GOD_DIM, PUB_DIM
    from fishrl.obs.encoder import OBS_DIM
    from fishrl.spaces.action_space import N
    assert local_dims() == (OBS_DIM, GOD_DIM, PUB_DIM, N)
    assert all(isinstance(v, int) for v in local_dims())
