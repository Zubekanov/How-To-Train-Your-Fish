"""Has the actor ALREADY learned the deckout clock internally?

Joseph's objection, and it is the right default: if a quantity is trivially computable from
the observation, the network can just learn it -- so feeding it in as a feature is redundant.

The counter-hypothesis is narrow and specific: the deckout clock is a PARITY function. Both
seats draw one card per turn from the SHARED library, so whoever is forced to draw from an
empty library is decided by (library_count mod 2) and whose draw is next -- and every extra
card drawn FLIPS it. But the observation carries library size as a single normalised float
(len(library)/80.0). Recovering N mod 2 from N/80 means fitting a 40-cycle square wave over
[0,1] and resolving 0.0125 steps. Parity from a smooth scalar is the textbook function ReLU
nets learn badly. "Computable" and "learnable from this encoding" are not the same claim.

This settles it directly instead of arguing. The actor's last layer is a plain Linear, so
anything the POLICY uses must be LINEARLY decodable from its penultimate activation. Probe it:

  hidden (256d, LEARNED)  -- if a linear probe recovers the clock, the net already has it and
                            the feature is redundant. Joseph is right.
  raw obs (ACTOR_IN)      -- what the net is given.
  naive  [lib/80, active] -- the encoding the net actually gets: a linear model cannot do
                            parity from a smooth scalar (expect ~chance).
  oracle [lib%2, drawer]  -- sanity: the same information, correctly encoded (expect ~1.0).

    python -m fishrl.eval.probe_deckout_clock --games 60
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from fishrl.train.belief_env import BeliefAugmentedEnv
from fishrl.train.checkpoint import load_checkpoint
from fishrl.train.collector import actor_act_fn
from fishrl.train.config import Config, resolve_device
from fishrl.train.scenarios import ScenarioEnv, get_scenario
from fishrl.train.train_loop import build_models, config_from_checkpoint, _load_model_state

NETS = ("actor", "critic", "guesser", "public")
_PRE_DRAW = ("untap", "upkeep", "draw", "")


def clock(g):
    """(label, naive_feats, oracle_feats).

    label = 1.0 iff p1 is the seat that runs out of draws FIRST under the naive schedule
    (one draw per turn each, no extra draws). Draw #(n+1) from now is the one that fails,
    and draws alternate starting with whoever draws next -- so the loser flips with the
    PARITY of the library count. Deterministic, public, and it decides ~81% of these games."""
    n = len(g.library)
    active = g.active_player if g.active_player in ("p1", "p2") else "p1"
    opp = "p2" if active == "p1" else "p1"
    drawn = g.current_step not in _PRE_DRAW          # the draw step precedes main1
    nxt = opp if drawn else active                   # who takes the NEXT draw
    loser = nxt if n % 2 == 0 else ("p2" if nxt == "p1" else "p1")
    label = 1.0 if loser == "p1" else 0.0
    naive = np.array([n / 80.0, float(active == "p1")], dtype=np.float32)
    oracle = np.array([float(n % 2), float(nxt == "p1")], dtype=np.float32)
    return label, naive, oracle


def collect(models, scenarios, n_games, seed, device, max_decisions=2000):
    hid, obs, naive, oracle, lab, gid = [], [], [], [], [], []
    trunk = models.actor.net[:-1]                    # everything before the final Linear
    act = actor_act_fn(models.actor)
    for name in scenarios:
        for i in range(n_games):
            senv = BeliefAugmentedEnv(
                models.guesser, belief=True,
                env=ScenarioEnv(get_scenario(name), max_decisions=max_decisions))
            senv.reset(seed=seed + i)
            for agent in senv.agent_iter(max_iter=max_decisions * 6):
                if senv.terminations[agent] or senv.truncations[agent]:
                    senv.step(None)
                    continue
                o = senv.observe(agent)
                g = senv.g
                x = torch.as_tensor(o["observation"], dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    h = trunk(x).squeeze(0).cpu().numpy()
                lb, nv, orc = clock(g)
                hid.append(h); obs.append(o["observation"].astype(np.float32))
                naive.append(nv); oracle.append(orc); lab.append(lb)
                gid.append(f'{name}:{i}')
                a, _ = act(o)
                senv.step(a)
        print(f"  [collect] {name}: {len(lab)} states", flush=True)
    return (np.stack(hid), np.stack(obs), np.stack(naive), np.stack(oracle),
            np.array(lab, dtype=np.float32), np.array(gid))


def probe(X, y, gid, device, hidden=(), steps=2500, batch=4096, seed=0):
    """Probe with a held-out split BY GAME. States inside a game have near-identical clock
    labels, so a state-level split leaks and inflates every high-dimensional arm (it is the
    same trap that faked a signal in the knowledge probe).

    `hidden=()` -> LINEAR, which is the meaningful test for the trunk: the actor's own final
    layer is Linear, so anything the POLICY acts on must be linearly readable there. A
    nonlinear probe answers a different question -- is the info merely PRESENT -- so both are
    reported."""
    games = np.unique(gid)
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    ho_games = set(games[:max(1, len(games) // 5)])
    ho = np.array([g in ho_games for g in gid])
    Xt, yt = torch.from_numpy(X[~ho]), torch.from_numpy(y[~ho])
    Xh, yh = torch.from_numpy(X[ho]), torch.from_numpy(y[ho])
    torch.manual_seed(seed)
    layers, d = [], X.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, 1))
    net = nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        b = torch.from_numpy(rng.integers(0, Xt.shape[0], size=min(batch, Xt.shape[0])))
        loss = lossf(net(Xt[b].to(device)).squeeze(-1), yt[b].to(device))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = torch.sigmoid(net(Xh.to(device)).squeeze(-1)).cpu()
    return float(((p > 0.5).float() == yh).float().mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="checkpoints/latest.pt")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seed", type=int, default=91_000)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    device = resolve_device(args.gpu)

    pl = load_checkpoint(args.ckpt, map_location="cpu")
    cd = pl["config"]
    cfg = config_from_checkpoint(cd)
    m = build_models(cfg)
    _load_model_state(m, pl["models"])
    if m.guesser is None or m.public is None:
        raise SystemExit("this tool interrogates the legacy guesser/public nets; "
                         "the given checkpoint is a v3 (bookkeeper/public-critic) "
                         "payload -- point it at a v1/v2 lineage instead")
    for net in (m.actor, m.guesser):
        net.eval(); net.to(device)
    assert cd["encoders"]["actor"] == "flat", "trunk probe assumes the flat actor"
    print(f"[clock] {args.ckpt}: iter={pl.get('done')} elapsed={pl.get('elapsed',0)/3600:.0f}h",
          flush=True)

    hid, obs, naive, oracle, y, gid = collect(
        m, ["deckout", "board_presence"], args.games, args.seed, device)
    print(f"[clock] {len(y)} states  |  base rate p1-decks-first = {y.mean():.3f}\n")

    maj = max(y.mean(), 1 - y.mean())
    print(f"{'probe on ...':40s} {'in_dim':>7} {'LINEAR':>8} {'MLP':>8}   (split BY GAME)")
    print(f"{'-- majority-class baseline --':40s} {'':>7} {maj:>8.3f} {maj:>8.3f}")
    for name, X in (
            ("actor TRUNK (256d, LEARNED)", hid),
            ("raw actor observation", obs),
            ("naive [lib/80, active] (what it gets)", naive),
            ("oracle [lib%2, next_drawer] (XOR!)", oracle)):
        lin = probe(X, y, gid, device)
        mlp = probe(X, y, gid, device, hidden=(64, 64))
        print(f"{name:40s} {X.shape[1]:>7} {lin:>8.3f} {mlp:>8.3f}", flush=True)

    print("\nRead: if the TRUNK probe is ~1.0 the policy already computes the clock -> the")
    print("feature is redundant (Joseph right). If it is near chance while ORACLE is ~1.0, the")
    print("information is present but the ENCODING hides it -> a computable feature the net")
    print("has NOT learned in 592h.")


if __name__ == "__main__":
    main()
