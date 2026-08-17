"""Probe: does "who knows what about the shared library" predict the WINNER?

RESULT (2026-07-14, on the 592h / 425k-iter checkpoint): **NO.** Tight, replicated null.
Kept as the record of a negative, and as the harness for the next such question.

The engine tracks knowledge state -- `LibrarySlot.known_by` (who has looked at which slot,
cleared on shuffle) and `CardInstance.known_by` (carried into hand on draw). It is PUBLIC
(casting Ponder is public, so "they know the top 3" is common knowledge; only the CONTENT is
private) but the actor's observation contains none of it. The question was whether closing
that gap is worth a checkpoint migration.

Design, and why each choice matters -- three earlier versions of this probe were WRONG:
  * Baseline is the trained **public estimator** (partial info), NOT the privileged critic.
    The critic already knows WHAT the top card is, so telling it "they have seen it" is a
    null question by construction. Knowledge asymmetry can only matter to a model that HAS
    uncertainty -- which is the actor's situation.
  * Baseline is the trained model's LOGIT, not raw features. Refitting an 8917-dim god critic
    on a probe-sized sample scored worse than a coin flip.
  * A **game-stage control** arm: a depleted library has been looked at more, so a naive
    knowledge signal can just be re-discovering the turn counter.
  * 5-fold CV split **by game** and a paired bootstrap resampled **by game**. Transitions in
    a game share a label: a transition-level split leaks it (it inflated `knowledge ALONE` to
    0.671 acc, which collapsed to the 0.587 base rate once split cleanly), and a per-sample
    bootstrap would claim ~65k independent draws when there are ~200.
  * Explicit **interaction terms** (`active x knows_top` = "the player about to draw already
    knows their draw") plus a nonlinear head, so "the model couldn't express it" is not an
    available excuse for a null.

Findings: `knowledge ALONE` sits exactly on the majority-class base rate under every clean
split, and the paired Brier gain over the stage control is -0.0003, 95% CI [-0.0020, +0.0012].
(The MLP head's significant-NEGATIVE CI is overfitting on ~200 effective games -- its
`knowledge ALONE` Brier is worse than a constant predictor -- not evidence that the channel
harms; read the linear head.)

    python -m fishrl.eval.probe_knowledge --ckpt checkpoints/latest.pt --games 200
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from fishrl.data.features import encode_god, encode_public
from fishrl.train.config import Config, resolve_device
from fishrl.train.checkpoint import load_checkpoint
from fishrl.train.belief_env import BeliefAugmentedEnv
from fishrl.train.collector import actor_act_fn
from fishrl.train.train_loop import build_models, config_from_checkpoint, _load_model_state

NETS = ("actor", "critic", "guesser", "public")
TOP_K = 10                      # library depth the knowledge channel resolves positionally
KNOW_DIM = 2 * TOP_K + 4 + 9    # raw bits + totals/hand-leak + DERIVED interactions


def knowledge_features(g) -> np.ndarray:
    """p1-oriented 'who knows what'. PUBLIC information: you saw them cast Ponder, so you
    know they know the top 3 -- you just don't know WHAT. Never encodes an identity here,
    only the fact of knowledge.

    The last 9 are DERIVED interaction terms. The tactically decisive quantity is not
    "p1 knows the top card", it is "the player ABOUT TO DRAW knows what they are drawing" --
    a product of (knows_top x is_active) that a linear model cannot form from the raw bits.
    Encoding it explicitly is what makes the probe capable of detecting the effect at all."""
    v = np.zeros(KNOW_DIM, dtype=np.float32)
    lib = g.library
    for i, slot in enumerate(lib[:TOP_K]):                  # positional: depth 0..K-1
        v[2 * i] = float(bool(slot.known_by.get("p1")))
        v[2 * i + 1] = float(bool(slot.known_by.get("p2")))
    o = 2 * TOP_K
    n = max(len(lib), 1)
    k1 = sum(1 for s in lib if s.known_by.get("p1")) / n     # how much of the deck each has seen
    k2 = sum(1 for s in lib if s.known_by.get("p2")) / n
    h1, h2 = g.players["p1"].hand, g.players["p2"].hand
    leak1 = sum(1 for i in h1 if "p2" in (g.objects[i].known_by or [])) / 12.0   # they see mine
    leak2 = sum(1 for i in h2 if "p1" in (g.objects[i].known_by or [])) / 12.0
    v[o + 0], v[o + 1], v[o + 2], v[o + 3] = k1, k2, leak1, leak2

    # ── derived: the interactions that actually decide plays ──────────────────
    top1 = float(bool(lib[0].known_by.get("p1"))) if lib else 0.0
    top2 = float(bool(lib[0].known_by.get("p2"))) if lib else 0.0
    t3_1 = sum(1 for s in lib[:3] if s.known_by.get("p1")) / 3.0
    t3_2 = sum(1 for s in lib[:3] if s.known_by.get("p2")) / 3.0
    a1 = float(g.active_player == "p1")
    a2 = float(g.active_player == "p2")
    d = o + 4
    v[d + 0] = a1
    v[d + 1] = a1 * top1        # the DRAWER knows their next draw (p1 to draw, p1 has seen it)
    v[d + 2] = a2 * top2        # ... same for p2
    v[d + 3] = a1 * t3_1
    v[d + 4] = a2 * t3_2
    v[d + 5] = top1 - top2      # signed top-card knowledge edge
    v[d + 6] = t3_1 - t3_2
    v[d + 7] = k1 - k2          # signed deck-knowledge edge
    v[d + 8] = leak2 - leak1    # signed hand-information edge (I see theirs minus they see mine)
    return v


def collect(models, n_games, seed, device, max_decisions=2000):
    """On-policy states from the CURRENT agent, with the winner back-filled per game.

    We record the TRAINED critic's own logit rather than raw god features. That critic is
    the real baseline -- it already sees every card identity and is well calibrated
    (priv_brier ~0.19 in production) -- and refitting an 8917-dim god critic from scratch on
    a probe-sized sample just overfits (it scored WORSE than a constant predictor). The
    question that actually matters is INCREMENTAL: does knowledge tell us anything the
    trained critic has not already extracted from the identities?"""
    benv = BeliefAugmentedEnv(models.guesser, belief=True, max_decisions=max_decisions)
    act = actor_act_fn(models.actor)
    priv, pub, know, stage, owner, gid = [], [], [], [], [], []
    for gi in range(n_games):
        benv.reset(seed=seed + gi)
        start = len(pub)
        for agent in benv.agent_iter(max_iter=max_decisions * 6):
            if benv.terminations[agent] or benv.truncations[agent]:
                benv.step(None)
                continue
            obs = benv.observe(agent)
            g = benv.g
            gt = torch.as_tensor(encode_god(g), dtype=torch.float32).unsqueeze(0).to(device)
            pt = torch.as_tensor(encode_public(g), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                priv.append(float(models.critic(gt).squeeze()))   # privileged (sees everything)
                pub.append(float(models.public(pt).squeeze()))    # PARTIAL info -- the actor's case
            know.append(knowledge_features(g))
            stage.append(stage_features(g))
            gid.append(gi)
            a, _ = act(obs)
            benv.step(a)
        w = benv.g.result.get("winner")
        owner.extend([1.0 if w == "p1" else 0.0] * (len(pub) - start))
        if (gi + 1) % 25 == 0:
            print(f"  [collect] {gi + 1}/{n_games} games  {len(pub)} transitions", flush=True)
    f32 = lambda xs: np.array(xs, dtype=np.float32)[:, None]
    return (f32(priv), f32(pub), np.stack(know), np.stack(stage),
            f32(owner).squeeze(-1), np.array(gid, dtype=np.int64))


def stage_features(g) -> np.ndarray:
    """Control arm: how far along is the game? The knowledge channel is CORRELATED with this
    (a depleted library has been looked at more), so without this control a knowledge signal
    could just be a game-stage proxy -- which every model already has."""
    return np.array([g.turn_number / 40.0,
                     len(g.library) / 80.0,
                     len(g.players["p1"].hand) / 12.0,
                     len(g.players["p2"].hand) / 12.0], dtype=np.float32)


def _mlp(din, hidden=()):
    """Linear (logistic) by default: the textbook incremental-information test. 25 inputs on
    50k+ samples cannot overfit, so a gain here is real information, not capacity."""
    layers, d = [], din
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, 1)]
    return nn.Sequential(*layers)


def _fit(xtr, ytr, device, steps=3000, batch=4096, seed=0, hidden=()):
    torch.manual_seed(seed)
    net = _mlp(xtr.shape[1], hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    M = xtr.shape[0]
    for _ in range(steps):
        idx = torch.from_numpy(rng.integers(0, M, size=min(batch, M)))
        xb, yb = xtr[idx].to(device), ytr[idx].to(device)
        loss = lossf(net(xb).squeeze(-1), yb)
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def _probs(net, x, device, batch=8192):
    out = []
    with torch.no_grad():
        for s in range(0, x.shape[0], batch):
            out.append(torch.sigmoid(net(x[s:s + batch].to(device)).squeeze(-1)).cpu())
    return torch.cat(out)


def kfold_se(X, Y, gid, device, steps, k=5, hidden=()):
    """Out-of-fold per-sample squared errors, folds split BY GAME (never by transition:
    transitions in a game share a label, so a transition-level split leaks it). Returns the
    OOF squared error and correctness for every sample -- so arms can be compared PAIRED."""
    games = np.unique(gid)
    rng = np.random.default_rng(0)
    rng.shuffle(games)
    folds = np.array_split(games, k)
    se = np.zeros(len(Y), dtype=np.float64)
    ok = np.zeros(len(Y), dtype=np.float64)
    for f in folds:
        ho = np.isin(gid, f)
        net = _fit(X[~ho], Y[~ho], device, steps=steps, hidden=hidden)
        p = _probs(net, X[ho], device)
        se[ho] = ((p - Y[ho]) ** 2).numpy()
        ok[ho] = ((p > 0.5).float() == Y[ho]).float().numpy()
    return se, ok


def boot_ci(diff, gid, n=4000, seed=0):
    """Paired bootstrap on the per-sample Brier difference, RESAMPLED BY GAME -- transitions
    within a game are near-perfectly correlated (same label), so a per-sample bootstrap would
    claim ~65k independent draws when there are only ~200."""
    games = np.unique(gid)
    idx = {g: np.flatnonzero(gid == g) for g in games}
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for b in range(n):
        pick = rng.choice(games, size=len(games), replace=True)
        means[b] = np.concatenate([diff[idx[g]] for g in pick]).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="checkpoints/latest.pt")
    ap.add_argument("--games", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1_234_000)
    ap.add_argument("--steps", type=int, default=1500)
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
        net.eval()
    print(f"[probe] {args.ckpt}: iter={pl.get('done')} elapsed={pl.get('elapsed', 0)/3600:.0f}h "
          f"| collecting {args.games} on-policy games ...", flush=True)

    for net in (m.critic, m.public):
        net.eval(); net.to(device)
    t = time.perf_counter()
    priv, pub, know, stage, y, gid = collect(m, args.games, args.seed, device)
    n = pub.shape[0]
    print(f"[probe] {n} transitions in {time.perf_counter()-t:.0f}s  "
          f"(knowledge={know.shape[1]} dims)  p1_win_rate={y.mean():.3f}", flush=True)

    P, U, K, S, Y = (torch.from_numpy(x) for x in (priv, pub, know, stage, y))

    def ref(logit, label):
        p = torch.sigmoid(logit).squeeze(-1)
        print(f"  {label:34s} Brier {float(((p - Y) ** 2).mean()):.4f}  "
              f"acc {float(((p > 0.5).float() == Y).float().mean()):.3f}")

    print()
    ref(P, "privileged critic (sees ALL cards)")
    ref(U, "public estimator (PARTIAL info)")
    print(f"  {'constant p=0.5':34s} Brier {float(((0.5 - Y) ** 2).mean()):.4f}\n")

    # The actor is a PARTIAL-information model, so the public estimator is the honest
    # baseline: knowledge asymmetry can only matter to a model that HAS uncertainty.
    arms = {
        "public (baseline)":          U,
        "public + stage (CONTROL)":   torch.cat([U, S], 1),
        "public + stage + knowledge": torch.cat([U, S, K], 1),
        "knowledge ALONE":            K,
        "stage ALONE (control)":      S,
    }
    # Run every arm under BOTH heads. The linear head is the clean incremental test; the MLP
    # can additionally form interactions the precomputed terms missed. If neither finds signal,
    # "the model couldn't express it" is no longer an available excuse.
    base, treat = "public + stage (CONTROL)", "public + stage + knowledge"
    verdicts = []
    for label, hidden in (("LINEAR head", ()), ("MLP head (64,64)", (64, 64))):
        print(f"\n--- {label} ---")
        print(f"{'arm':28s} {'in_dim':>7} {'Brier':>8} {'acc':>7}   (5-fold OOF, split by GAME)")
        se, acc = {}, {}
        for name, X in arms.items():
            s_, a_ = kfold_se(X, Y, gid, device, args.steps, hidden=hidden)
            se[name], acc[name] = s_, a_
            print(f"{name:28s} {X.shape[1]:>7} {s_.mean():>8.4f} {a_.mean():>7.3f}", flush=True)
        diff = se[base] - se[treat]                  # >0 means knowledge REDUCES Brier
        lo, hi = boot_ci(diff, gid)                  # bootstrap BY GAME (n_eff ~= #games)
        print(f"  paired Brier gain from knowledge : {diff.mean():+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"  accuracy gain                    : {acc[treat].mean() - acc[base].mean():+.4f}")
        verdicts.append(lo > 0)
    print("\nVERDICT:", "REAL -- a CI excludes zero on the positive side. Expose it to the actor."
          if any(verdicts) else
          "NOT PROVEN -- no head shows a positive gain whose CI excludes zero. Do not migrate.")


if __name__ == "__main__":
    main()
