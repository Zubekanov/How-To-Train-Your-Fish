"""Does the CURRENT policy actually use the deckout clock? Re-measure, don't cite.

The "clock is provably ignored (behaviour Δ0.000)" result predates the v2 restart:
it was measured on the flat v1 lineage right after the feature shipped. The v2
entity actor has since trained hundreds of hours WITH the clock bits present, so
the claim needs re-measuring, not repeating. Three instruments:

  1. --probe: state-level sensitivity. Over mirror self-play + deckout +
     board_presence states, compare the actor's masked action distribution with
     the clock bits (a) intact, (b) zeroed, (c) counterfactually FLIPPED (parity
     and decks_first inverted — exactly the world where the library count is ±1).
     TV distance + argmax-change rate, binned by library size. Same flip applied
     to the critic's god clock: |ΔP(p1 wins)| and its DIRECTION (flipping
     p1-decks-first off should raise P(p1) if the critic reads it correctly).
  2. --arm on|zero|flip: the behavioural bottom line. Win-rate on the deckout and
     board_presence scenarios vs the engine's heuristic_1_3, with the clock intact,
     zeroed, or adversarially flipped at inference — identical seed lists across
     arms, so the comparison is paired. If the policy ignores the clock, all three
     arms are statistically identical.
  3. --summary d1.json d2.json ...: combine arm JSONs into the paired verdict.

    python -m fishrl.eval.clock_usage --probe
    python -m fishrl.eval.clock_usage --arm on   --out arm_on.json
    python -m fishrl.eval.clock_usage --arm zero --out arm_zero.json
    python -m fishrl.eval.clock_usage --arm flip --out arm_flip.json
    python -m fishrl.eval.clock_usage --summary arm_on.json arm_zero.json arm_flip.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from fishrl.data.features import GOD_DIM, encode_god
from fishrl.obs.encoder import OBS_DIM
from fishrl.train.belief_env import BeliefAugmentedEnv
from fishrl.train.checkpoint import load_checkpoint
from fishrl.train.scenarios import ScenarioEnv, get_scenario
from fishrl.train.train_loop import build_models, config_from_checkpoint, _load_model_state

C0 = OBS_DIM - 3          # clock = last 3 dims of the perspective block (belief rides after)
G0 = GOD_DIM - 3          # clock = last 3 dims of the god vector (p1-oriented)
LIB_BINS = ((41, 80), (21, 40), (11, 20), (0, 10))


def load_models(ckpt: str):
    pl = load_checkpoint(ckpt, map_location="cpu")
    cfg = config_from_checkpoint(pl["config"])
    m = build_models(cfg)
    _load_model_state(m, pl["models"])
    for net in (m.actor, m.guesser, m.critic):
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
    print(f"[clock_usage] {ckpt}: iter={pl.get('done')} elapsed={pl.get('elapsed', 0) / 3600:.0f}h "
          f"actor={pl['config']['encoders']['actor']}", flush=True)
    return m


def apply_mode(obs: np.ndarray, mode: str) -> np.ndarray:
    """Return a copy of the actor input with the clock bits transformed."""
    if mode == "on":
        return obs
    out = obs.copy()
    if mode == "zero":
        out[..., C0:C0 + 3] = 0.0
    elif mode == "flip":                      # the library-count-±1 counterfactual:
        out[..., C0] = 1.0 - out[..., C0]     # parity inverts,
        out[..., C0 + 2] = 1.0 - out[..., C0 + 2]  # so the decks-first seat inverts;
    return out                                # next-drawer (bit 1) is unchanged


def _sample(actor, obs, mask):
    x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    mk = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
    logp = actor.log_probs(x, mk)[0]
    return int(torch.multinomial(logp.exp(), 1))


# ── 1. state-level sensitivity probe ─────────────────────────────────────────

def collect_states(m, source: str, n_games: int, seed: int, max_decisions=2000):
    """Play games with the CURRENT policy; record (obs, mask, lib, clock_defined,
    god) at every learner decision. Mirror: both seats are learner decisions."""
    obs_l, mask_l, lib_l, def_l, god_l = [], [], [], [], []
    for i in range(n_games):
        if source == "mirror":
            senv = BeliefAugmentedEnv(m.guesser, belief=True, max_decisions=max_decisions)
        else:
            senv = BeliefAugmentedEnv(m.guesser, belief=True,
                                      env=ScenarioEnv(get_scenario(source),
                                                      max_decisions=max_decisions))
        senv.reset(seed=seed + i)
        torch.manual_seed(seed + i)
        for agent in senv.agent_iter(max_iter=max_decisions * 6):
            if senv.terminations[agent] or senv.truncations[agent]:
                senv.step(None)
                continue
            o = senv.observe(agent)
            g = senv.g
            obs_l.append(o["observation"].astype(np.float16))
            mask_l.append(o["action_mask"].astype(np.int8))
            lib_l.append(len(g.library))
            def_l.append(g.active_player in ("p1", "p2"))
            god_l.append(encode_god(g).astype(np.float16))
            senv.step(_sample(m.actor, o["observation"], o["action_mask"]))
    print(f"  [collect] {source}: {len(lib_l)} states / {n_games} games", flush=True)
    return (np.stack(obs_l), np.stack(mask_l), np.asarray(lib_l),
            np.asarray(def_l), np.stack(god_l))


def _batched_probs(actor, obs16, mask, mode, batch=1024):
    out = []
    for i in range(0, len(obs16), batch):
        o = apply_mode(obs16[i:i + batch].astype(np.float32), mode)
        x = torch.from_numpy(o)
        mk = torch.from_numpy(mask[i:i + batch].astype(np.float32))
        out.append(actor.log_probs(x, mk).exp().numpy())
    return np.concatenate(out)


def probe(m, args):
    rows = []
    for source, n in (("mirror", args.probe_mirror_games),
                      ("deckout", args.probe_scenario_games),
                      ("board_presence", args.probe_scenario_games)):
        obs, mask, lib, dfn, god = collect_states(m, source, n, args.seed)
        p_on = _batched_probs(m.actor, obs, mask, "on")
        p_zero = _batched_probs(m.actor, obs, mask, "zero")
        p_flip = _batched_probs(m.actor, obs, mask, "flip")

        with torch.no_grad():
            v_on, v_flip = [], []
            for i in range(0, len(god), 1024):
                gvec = god[i:i + 1024].astype(np.float32)
                v_on.append(m.critic.p1_winprob(torch.from_numpy(gvec)).numpy())
                gflip = gvec.copy()
                gflip[:, G0] = 1.0 - gflip[:, G0]
                gflip[:, G0 + 2] = 1.0 - gflip[:, G0 + 2]
                v_flip.append(m.critic.p1_winprob(torch.from_numpy(gflip)).numpy())
        v_on, v_flip = np.concatenate(v_on), np.concatenate(v_flip)

        tv_zero = 0.5 * np.abs(p_on - p_zero).sum(1)
        tv_flip = 0.5 * np.abs(p_on - p_flip).sum(1)
        am_zero = (p_on.argmax(1) != p_zero.argmax(1))
        am_flip = (p_on.argmax(1) != p_flip.argmax(1))
        dv = v_flip - v_on
        # direction: flipping sends decks_first 1->0 (good for p1) or 0->1 (bad)
        was_df = god[:, G0 + 2].astype(np.float32) > 0.5

        print(f"\n[{source}] actor sensitivity (n={len(lib)}; flip rows are "
              f"clock-defined states only):")
        print(f"  {'library':>9} {'n':>6} {'TV zero':>8} {'TV flip':>8} "
              f"{'argmaxΔ zero':>12} {'argmaxΔ flip':>12} {'critic |ΔP| flip':>16}")
        for lo, hi in LIB_BINS:
            mb = (lib >= lo) & (lib <= hi)
            mf = mb & dfn
            if mb.sum() == 0:
                continue
            print(f"  {f'{lo}-{hi}':>9} {mb.sum():>6} {tv_zero[mb].mean():>8.4f} "
                  f"{tv_flip[mf].mean() if mf.any() else float('nan'):>8.4f} "
                  f"{am_zero[mb].mean():>12.4f} "
                  f"{am_flip[mf].mean() if mf.any() else float('nan'):>12.4f} "
                  f"{np.abs(dv[mf]).mean() if mf.any() else float('nan'):>16.4f}")
            rows.append({"source": source, "bin": f"{lo}-{hi}", "n": int(mb.sum()),
                         "tv_zero": float(tv_zero[mb].mean()),
                         "tv_flip": float(tv_flip[mf].mean()) if mf.any() else None,
                         "argmax_zero": float(am_zero[mb].mean()),
                         "argmax_flip": float(am_flip[mf].mean()) if mf.any() else None,
                         "critic_absdp": float(np.abs(dv[mf]).mean()) if mf.any() else None})
        for label, mrow in (("was decks-first=1 -> 0 (good for p1)", was_df & dfn),
                            ("was decks-first=0 -> 1 (bad for p1)", (~was_df) & dfn)):
            if mrow.any():
                print(f"  critic signed ΔP | {label}: {dv[mrow].mean():+.4f} (n={mrow.sum()})")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"\n[probe] wrote {args.out}")


# ── 2. win-rate arms ─────────────────────────────────────────────────────────

def play_arm(m, mode: str, args) -> dict:
    res = {"mode": mode, "seed": args.seed, "n": args.wr_games, "scenarios": {}}
    for source in ("deckout", "board_presence"):
        t0 = time.time()
        outcomes = []
        for i in range(args.wr_games):
            senv = BeliefAugmentedEnv(m.guesser, belief=True,
                                      env=ScenarioEnv(get_scenario(source),
                                                      max_decisions=2000))
            senv.reset(seed=args.seed + i)
            torch.manual_seed(args.seed + i)
            for agent in senv.agent_iter(max_iter=2000 * 6):
                if senv.terminations[agent] or senv.truncations[agent]:
                    senv.step(None)
                    continue
                o = senv.observe(agent)
                senv.step(_sample(m.actor, apply_mode(o["observation"], mode),
                                  o["action_mask"]))
            w = senv.winner
            outcomes.append(1 if w == "p1" else (0 if w == "p2" else -1))
            if (i + 1) % 50 == 0:
                print(f"  [{mode}/{source}] {i + 1}/{args.wr_games} "
                      f"({(time.time() - t0) / (i + 1) * 1000:.0f}ms/game)", flush=True)
        wins = sum(1 for o in outcomes if o == 1)
        res["scenarios"][source] = {"outcomes": outcomes, "wr": wins / len(outcomes)}
        print(f"[{mode}] {source}: wr={wins / len(outcomes):.3f} "
              f"({wins}/{len(outcomes)}, draws={sum(1 for o in outcomes if o == -1)})",
              flush=True)
    return res


def summary(paths):
    arms = {a["mode"]: a for a in (json.load(open(p)) for p in paths)}
    base = arms.get("on")
    for source in base["scenarios"]:
        print(f"\n== {source} (n={base['n']}, paired seeds) ==")
        oc_on = base["scenarios"][source]["outcomes"]
        for mode, a in arms.items():
            oc = a["scenarios"][source]["outcomes"]
            wr = a["scenarios"][source]["wr"]
            k = sum(1 for o in oc if o == 1)
            n = len(oc)
            z = 1.96
            ph = k / n
            d = 1 + z * z / n
            c = (ph + z * z / (2 * n)) / d
            h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
            extra = ""
            if mode != "on":
                flips_lw = sum(1 for x, y in zip(oc_on, oc) if x == 1 and y != 1)
                flips_wl = sum(1 for x, y in zip(oc_on, oc) if x != 1 and y == 1)
                extra = (f"  vs on: Δwr={wr - base['scenarios'][source]['wr']:+.3f} "
                         f"(paired flips: on-won-this-lost {flips_lw}, "
                         f"on-lost-this-won {flips_wl})")
            print(f"  {mode:>5}: wr={wr:.3f} [{c - h:.3f}, {c + h:.3f}]{extra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="checkpoints-v2/latest.pt")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe-mirror-games", type=int, default=16)
    ap.add_argument("--probe-scenario-games", type=int, default=50)
    ap.add_argument("--arm", choices=("on", "zero", "flip"))
    ap.add_argument("--wr-games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=5_000_000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", nargs="+", default=None)
    args = ap.parse_args()

    if args.summary:
        summary(args.summary)
        return 0
    torch.set_num_threads(4)
    m = load_models(args.ckpt)
    if args.probe:
        probe(m, args)
    if args.arm:
        res = play_arm(m, args.arm, args)
        out = args.out or f"clock_arm_{args.arm}.json"
        json.dump(res, open(out, "w"), indent=1)
        print(f"[arm] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
