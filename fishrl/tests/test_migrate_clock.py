"""The clock warm-start must be BEHAVIOUR-PRESERVING: 592h of training is the thing at risk.

The clock is 3 scalars appended to each encoder's globals, so every old input still exists at a
computable new index. Widen the first Linear, splice the old columns into their new positions,
zero the new ones -> the migrated net is functionally identical at step 0 and only then starts
learning to use the channel.
"""
import numpy as np
import torch
import torch.nn as nn

from fishrl.obs.encoder import _CLOCK
from fishrl.train.migrate_clock import GROWN, _grow_rows, _strip_clock, migrate_optim


def test_grow_rows_preserves_the_function_exactly():
    """A widened Linear fed the widened input must produce the SAME output as the original fed
    the original input -- that is the whole warm-start guarantee, in one assertion."""
    torch.manual_seed(0)
    old_in, new_in, at = 11, 11 + _CLOCK, 8      # clock spliced into the MIDDLE (actor layout)
    lin = nn.Linear(old_in, 5)
    w2 = _grow_rows(lin.weight.data, old_in, new_in, at, _CLOCK)
    big = nn.Linear(new_in, 5)
    big.weight.data, big.bias.data = w2, lin.bias.data.clone()

    x_old = torch.randn(7, old_in)
    x_new = torch.zeros(7, new_in)
    x_new[:, :at] = x_old[:, :at]
    x_new[:, at + _CLOCK:] = x_old[:, at:]
    x_new[:, at:at + _CLOCK] = torch.randn(7, _CLOCK)      # ARBITRARY clock values...
    # ...must not matter: the new columns are zero, so they contribute nothing.
    assert torch.allclose(big(x_new), lin(x_old), atol=1e-6)
    assert (w2[:, at:at + _CLOCK] == 0).all()


def test_strip_clock_inverts_the_splice():
    v = np.arange(20, dtype=np.float32)
    assert _strip_clock(v, 8).tolist() == [*range(8), *range(8 + _CLOCK, 20)]


def test_migrate_optim_widens_the_entity_critic_moments_too():
    """Regression: matching Adam moments by encoder width MISSES the entity critic, whose first
    Linear sees enc_dim (~1053), not GOD_DIM (~8917). Left un-widened it does not fail at load --
    it fails on the first exp_avg.add_(grad), minutes into the branch."""
    GROWN.clear()
    GROWN[(512, 1053)] = (1053 + _CLOCK, 1053)          # as recorded by migrate_critic
    GROWN[(256, 6514)] = (6514 + _CLOCK, 6494)          # as recorded by migrate_actor
    optim = {"ppo": {"state": {
        0: {"exp_avg": torch.zeros(256, 6514), "exp_avg_sq": torch.ones(256, 6514)},
        1: {"exp_avg": torch.zeros(512, 1053), "exp_avg_sq": torch.ones(512, 1053)},
        2: {"exp_avg": torch.zeros(64)},                 # 1-D: left alone
    }}}
    out = migrate_optim(optim)
    st = out["ppo"]["state"]
    assert st[0]["exp_avg"].shape == (256, 6514 + _CLOCK)
    assert st[1]["exp_avg"].shape == (512, 1053 + _CLOCK)     # the critic -- the one that was missed
    assert st[1]["exp_avg_sq"].shape == (512, 1053 + _CLOCK)
    assert st[2]["exp_avg"].shape == (64,)                    # untouched
    # the spliced-in moment columns must be ZERO (no stale second moment on new columns)
    assert (st[1]["exp_avg_sq"][:, 1053:] == 0).all()
