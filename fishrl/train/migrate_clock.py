"""Warm-start a pre-clock checkpoint onto the deckout-clock observation.

The clock adds 3 scalars to the TAIL of the globals block of each encoder (perspective, god,
public). Every pre-existing input therefore still exists, at a computable new index -- so the
old weights can be remapped exactly and the 3 new input columns zero-initialised. At step 0 the
migrated agent is FUNCTIONALLY IDENTICAL to the checkpoint it came from; it then learns to use
the channel. 592 hours of training are preserved: this BRANCHES the run, it does not restart it.

What has to move:
  * actor / guesser / public : flat MLPs -- first Linear gains input columns.
  * critic                   : entity encoder -- globals are appended verbatim after the pooled
                               zones (`globals_dim = total_in - R*CARD_F`), so its first Linear
                               gains columns at the TAIL, and the CardEncoder is untouched.
  * Adam moments             : exp_avg / exp_avg_sq for those same rows, or the optimiser
                               carries stale second-moment estimates on shifted columns.
  * the frozen-self anchor and the PFSP past-self ring: they consume observations too.

The gate: for a batch of real states, old_actor(old_obs) must equal new_actor(new_obs) within
float tolerance. If that identity fails we abort rather than corrupt the run. Nothing is
overwritten -- the migrated checkpoint is written to a NEW path.

    python -m fishrl.train.migrate_clock --in checkpoints/latest.pt --out checkpoints/clock.pt
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import torch

# NB: these are the NEW dims (this module is imported after the feature landed).
from fishrl.obs.encoder import OBS_DIM, _CLOCK, _GLOBALS
from fishrl.data.features import GOD_DIM, PUB_DIM
from fishrl.obs import vocab as V

OLD_OBS = OBS_DIM - _CLOCK
OLD_GOD = GOD_DIM - _CLOCK
OLD_PUB = PUB_DIM - _CLOCK
ACTOR_IN, OLD_ACTOR_IN = OBS_DIM + V.N_NAMES, OLD_OBS + V.N_NAMES


def _grow_rows(w: torch.Tensor, old_in: int, new_in: int, insert_at: int, n_new: int):
    """Widen a Linear's input: copy the old columns into their new positions, ZERO the new ones.

    The clock sits at the tail of the GLOBALS block, which is itself the tail of the encoder
    vector -- but the actor's input is `obs ⊕ guess`, so for the actor the new columns land in
    the MIDDLE (before the guess block), not at the end. Hence a general splice rather than a
    concat: `insert_at` is where the new columns go."""
    assert w.shape[1] == old_in, (w.shape, old_in)
    out = w.new_zeros((w.shape[0], new_in))
    out[:, :insert_at] = w[:, :insert_at]                       # everything before the clock
    out[:, insert_at + n_new:] = w[:, insert_at:]               # everything after it, shifted
    return out                                                   # the n_new columns stay ZERO


# (out_features, old_in) -> (new_in, insert_at) for every Linear we actually widened. Adam's
# state is keyed by parameter POSITION, not name, so the moments can only be matched back by
# shape -- and guessing the widths from the encoder dims is wrong for the entity critic, whose
# first Linear sees enc_dim (1053), NOT GOD_DIM. Recording what we grew is the only safe way.
GROWN: dict = {}


def _migrate_linear(sd: dict, key: str, old_in: int, new_in: int, insert_at: int):
    GROWN[(sd[key].shape[0], old_in)] = (new_in, insert_at)
    sd[key] = _grow_rows(sd[key], old_in, new_in, insert_at, new_in - old_in)


def _first_linear(state: dict) -> str:
    """Key of the first Linear's weight in a flat MLP state_dict ('net.0.weight')."""
    for k in state:
        if k.endswith("net.0.weight"):
            return k
    raise KeyError(f"no net.0.weight in {list(state)[:6]}")


def migrate_actor(state: dict) -> dict:
    # actor input = perspective(OBS_DIM) ⊕ guess(N_NAMES): the clock goes at the END of the
    # OBSERVATION, i.e. BEFORE the guess block -> a mid-vector splice.
    k = _first_linear(state)
    _migrate_linear(state, k, OLD_ACTOR_IN, ACTOR_IN, insert_at=OLD_OBS)
    return state


def migrate_guesser(state: dict) -> dict:
    # guesser(persp, prev_guess): same layout as the actor (persp ⊕ prev_guess).
    k = _first_linear(state)
    _migrate_linear(state, k, OLD_ACTOR_IN, ACTOR_IN, insert_at=OLD_OBS)
    return state


def migrate_public(state: dict) -> dict:
    k = _first_linear(state)
    _migrate_linear(state, k, OLD_PUB, PUB_DIM, insert_at=OLD_PUB)   # clock is the tail
    return state


def migrate_critic(state: dict) -> dict:
    """Entity encoder: globals are concatenated AFTER the pooled zones, so the encoder's output
    grows by _CLOCK at its tail -> the trunk's first Linear gains columns at its tail. The card
    encoder / zone / pos embeddings are all untouched (rows did not change)."""
    k = _first_linear(state)
    old_in = state[k].shape[1]
    _migrate_linear(state, k, old_in, old_in + _CLOCK, insert_at=old_in)
    return state


MIGRATORS = {"actor": migrate_actor, "critic": migrate_critic,
             "guesser": migrate_guesser, "public": migrate_public}


def migrate_models(models: dict) -> dict:
    return {name: MIGRATORS[name](copy.deepcopy(sd)) if name in MIGRATORS else sd
            for name, sd in models.items()}


def migrate_optim(optim: dict) -> dict:
    """Adam keeps exp_avg / exp_avg_sq per PARAMETER TENSOR, indexed positionally. Every weight
    we widened must have its moments widened IDENTICALLY, or the first step dies on a shape
    mismatch (and if it did not, Adam would normalise the new columns' gradients by a stale
    second moment). Matched via GROWN, so the entity critic -- whose first Linear sees enc_dim,
    not GOD_DIM -- is not missed. Must be called AFTER migrate_models has populated GROWN."""
    assert GROWN, "migrate_optim called before migrate_models"
    if not optim:
        return optim
    out = copy.deepcopy(optim)
    grown = 0
    for opt in (out.values() if isinstance(out, dict) else []):
        for slot in ((opt or {}).get("state") or {}).values():
            for mkey in ("exp_avg", "exp_avg_sq"):
                t = slot.get(mkey)
                if not isinstance(t, torch.Tensor) or t.dim() != 2:
                    continue
                hit = GROWN.get((t.shape[0], t.shape[1]))
                if hit is not None:
                    new_in, at = hit
                    slot[mkey] = _grow_rows(t, t.shape[1], new_in, at, new_in - t.shape[1])
                    grown += 1
    print(f"[migrate] widened {grown} Adam moment tensors ({sorted(GROWN)})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", default="checkpoints/latest.pt")
    ap.add_argument("--out", dest="dst", default="checkpoints/clock.pt")
    ap.add_argument("--games", type=int, default=6, help="games of states for the identity gate")
    args = ap.parse_args()

    from fishrl.train.checkpoint import load_checkpoint, save_checkpoint

    pl = load_checkpoint(args.src, map_location="cpu")
    print(f"[migrate] {args.src}: iter={pl.get('done')} elapsed={pl.get('elapsed',0)/3600:.0f}h")
    print(f"[migrate] OBS {OLD_OBS}->{OBS_DIM}  GOD {OLD_GOD}->{GOD_DIM}  PUB {OLD_PUB}->{PUB_DIM}")

    old_models = copy.deepcopy(pl["models"])
    pl["models"] = migrate_models(pl["models"])
    if pl.get("frozen"):
        pl["frozen"] = migrate_models(pl["frozen"])
    if pl.get("optim"):
        pl["optim"] = migrate_optim(pl["optim"])
    lg = pl.get("league") or {}
    for s in lg.get("selves", []):                       # the PFSP past-self ring acts on obs too
        s["actor"] = migrate_actor(s["actor"])
        s["guesser"] = migrate_guesser(s["guesser"])

    from fishrl.train.config import Config
    cd = pl["config"]
    cfg = Config(seed=cd["seed"], use_belief=cd.get("use_belief", True),
                 critic_hidden=tuple(cd.get("critic_hidden", (512, 512, 256))),
                 **{f"{n}_encoder": cd["encoders"][n] for n in MIGRATORS})
    if not _identity_gate(old_models, pl["models"], cfg, args.games):
        raise SystemExit("[migrate] ABORTED: identity gate failed -- checkpoint NOT written")
    if not _optim_gate(pl, cfg):
        raise SystemExit("[migrate] ABORTED: optimiser gate failed -- checkpoint NOT written")
    save_checkpoint(args.dst, pl)
    print(f"[migrate] wrote {args.dst}  (source untouched -- this BRANCHES the run)")


def _strip_clock(v: np.ndarray, at: int) -> np.ndarray:
    """The pre-clock vector: the same features with the 3 clock columns deleted. The old
    ENCODER no longer exists (it was edited in place), but it does not need to -- the clock is
    the tail of the globals, which is the tail of the vector, so deleting those columns
    reconstructs exactly what the old encoder produced."""
    return np.delete(v, np.arange(at, at + _CLOCK), axis=-1)


def _identity_gate(old_models: dict, new_models: dict, cfg, n_games: int) -> bool:
    """THE safety property: on real states, the migrated nets must produce the SAME outputs as
    the originals. The new columns are zero, so the clock contributes nothing at step 0 -- the
    592h policy is preserved exactly, and only then starts learning to use the channel.

    Rebuilds the OLD nets at the OLD input widths (the model classes now read the NEW dims from
    the module, so they cannot be constructed directly) and feeds them the clock-stripped
    vectors."""
    from fishrl.data.features import GOD_SLOTS, encode_god, encode_public
    from fishrl.env.aec_env import FishAEC
    from fishrl.models.entity_encoder import build_entity_encoder, layout_from_slots
    from fishrl.models.mlp import make_mlp
    from fishrl.obs.encoder import encode_observation
    from fishrl.spaces import action_space as A
    from fishrl.train.train_loop import build_models, _load_model_state

    new = build_models(cfg)
    _load_model_state(new, new_models)
    for n in (new.actor, new.critic, new.guesser, new.public):
        n.eval()

    def _sub(sd, pre="net."):
        return {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}

    old_actor = make_mlp(OLD_ACTOR_IN, A.N, cfg.hidden)
    old_actor.load_state_dict(_sub(old_models["actor"]))
    old_public = make_mlp(OLD_PUB, 1, cfg.hidden)
    old_public.load_state_dict(_sub(old_models["public"]))
    old_enc = build_entity_encoder("entity", *layout_from_slots(GOD_SLOTS), OLD_GOD)
    old_enc.load_state_dict(_sub(old_models["critic"], "enc."))
    old_critic_net = make_mlp(old_enc.enc_dim, 1, cfg.critic_hidden)
    old_critic_net.load_state_dict(_sub(old_models["critic"]))
    for n in (old_actor, old_public, old_enc, old_critic_net):
        n.eval()

    zg = np.zeros(V.N_NAMES, dtype=np.float32)
    worst = {"actor": 0.0, "critic": 0.0, "public": 0.0}
    rng = np.random.default_rng(0)
    checked = 0
    for gi in range(n_games):
        env = FishAEC(max_decisions=400)
        env.reset(seed=1234 + gi)
        while env.agents:
            s = env.agent_selection
            if env.terminations[s] or env.truncations[s]:
                env.step(None)
                continue
            g = env.g
            obs = encode_observation(g, s)
            x_new = np.concatenate([obs, zg]).astype(np.float32)
            x_old = _strip_clock(x_new, OLD_OBS)          # clock sits before the guess block
            god_new, pub_new = encode_god(g), encode_public(g)
            with torch.no_grad():
                a_new = new.actor(torch.from_numpy(x_new).unsqueeze(0))
                a_old = old_actor(torch.from_numpy(x_old).unsqueeze(0))
                c_new = new.critic(torch.from_numpy(god_new).unsqueeze(0))
                c_old = old_critic_net(old_enc(
                    torch.from_numpy(_strip_clock(god_new, OLD_GOD)).unsqueeze(0))).squeeze(-1)
                p_new = new.public(torch.from_numpy(pub_new).unsqueeze(0))
                p_old = old_public(torch.from_numpy(
                    _strip_clock(pub_new, OLD_PUB)).unsqueeze(0)).squeeze(-1)
            worst["actor"] = max(worst["actor"], float((a_new - a_old).abs().max()))
            worst["critic"] = max(worst["critic"], float((c_new - c_old).abs().max()))
            worst["public"] = max(worst["public"], float((p_new - p_old).abs().max()))
            checked += 1
            m = env.observe(s)["action_mask"]
            env.step(int(rng.choice(np.flatnonzero(m))))

    tol = 1e-4
    ok = all(v < tol for v in worst.values())
    print(f"[migrate] gate over {checked} real states -- max |new - old|:")
    for k, v in worst.items():
        print(f"           {k:8s} {v:.2e}   {'OK' if v < tol else 'FAIL'}")
    return ok



def _optim_gate(pl: dict, cfg) -> bool:
    """Rebuild the optimisers from the migrated state and take ONE real step.

    This is the gate that matters for the Adam moments: a moment tensor left at the old width
    does not fail at load time -- it fails on the first `exp_avg.mul_(b).add_(grad)`, i.e. some
    minutes into the branch. Catching it here (and not there) is the whole point."""
    import torch as T

    from fishrl.train.train_loop import build_models, _load_model_state

    m = build_models(cfg)
    _load_model_state(m, pl["models"])
    opt_ppo = T.optim.Adam(list(m.actor.parameters()) + list(m.critic.parameters()), lr=cfg.lr_ppo)
    opt_g = T.optim.Adam(m.guesser.parameters(), lr=cfg.lr_guesser)
    opt_p = T.optim.Adam(m.public.parameters(), lr=cfg.lr_public)
    saved = pl.get("optim") or {}
    named = {"ppo": opt_ppo, "guesser": opt_g, "public": opt_p}
    try:
        for k, o in named.items():
            if k in saved:
                o.load_state_dict(saved[k])
        for net, o in ((m.actor, opt_ppo), (m.guesser, opt_g), (m.public, opt_p)):
            loss = sum(p.sum() for p in net.parameters())      # any grad; we only need shapes
            o.zero_grad(); loss.backward(); o.step()
    except (RuntimeError, ValueError) as e:
        print(f"[migrate] optimiser gate FAILED: {e}")
        return False
    print(f"[migrate] optimiser gate: loaded {sorted(saved)} and stepped -- OK")
    return True

if __name__ == "__main__":
    main()
