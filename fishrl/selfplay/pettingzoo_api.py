"""PettingZoo-conformant factory for the Forgetful Fish environment.

``raw_env`` is the bare :class:`FishAEC`; ``env`` applies the standard PettingZoo
wrappers (assert-out-of-bounds + order enforcing) for safe interactive use.
"""
from __future__ import annotations

from pettingzoo.utils import wrappers

from fishrl.env.aec_env import FishAEC


def raw_env(**kwargs) -> FishAEC:
    return FishAEC(**kwargs)


def env(**kwargs):
    e = raw_env(**kwargs)
    e = wrappers.AssertOutOfBoundsWrapper(e)
    e = wrappers.OrderEnforcingWrapper(e)
    return e
