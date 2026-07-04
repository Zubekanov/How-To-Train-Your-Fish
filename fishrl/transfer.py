"""Portable export/import of a training run — move progress between machines.

    python -m fishrl.transfer export --ckpt-dir checkpoints
    python -m fishrl.transfer export --ckpt-dir checkpoints --out run.zip --with-archives
    python -m fishrl.transfer import --zip run.zip --ckpt-dir checkpoints [--force] [--merge-stats]
    python -m fishrl.transfer claim --ckpt-dir checkpoints

Export/import double as the RELAY handoff (see fishrl.train.ownership): export
refuses under a live trainer and releases local ownership once the zip is cut;
import refuses a stale-generation zip and claims ownership for this host;
``claim`` is the explicit human recovery when a handoff was interrupted between
the two (source released, destination never imported -- nobody owns the turn).

The zip carries everything a run needs to continue elsewhere: ``latest.pt``
(the fully self-contained resume checkpoint: models, frozen anchor, optimizer
states, RNG, counters, and the whole PFSP league), the ``stats.json``
time-series, the eval service's ``best.pt``/``best.json``, and a
``manifest.json`` describing the source (git commit, torch version, platform,
iteration, per-file sha256). ``archive_*.pt`` / ``step_*.pt`` ride along only
behind flags — archives grow forever.

Round-trip notes: checkpoints are saved/loaded with ``map_location`` and the
CUDA RNG entry is applied only where CUDA exists, so CPU↔GPU moves are safe.
Torch-version skew is the risky axis (newer-saved into older-loaded);
``import`` warns when the manifest's torch minor differs from the local one —
keep both machines inside the requirements.txt pin. Export with the trainer
stopped for a clean cut, or accept that ``latest.pt`` is whatever the last
15-minute checkpoint wrote (reads are safe: writes are atomic).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zipfile

MANIFEST = "manifest.json"
SCHEMA = 1
_CORE = ("stats.json", "best.pt", "best.json")         # optional companions to latest.pt


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10,
                             cwd=os.path.dirname(os.path.abspath(__file__)))
        return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _members(ckpt_dir: str, with_archives: bool, with_milestones: bool) -> list:
    names = ["latest.pt"] + [n for n in _CORE if os.path.exists(os.path.join(ckpt_dir, n))]
    if with_archives:
        names += sorted(os.path.basename(p)
                        for p in glob.glob(os.path.join(ckpt_dir, "archive_*.pt")))
    if with_milestones:
        names += sorted(os.path.basename(p)
                        for p in glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
    return names


def export(ckpt_dir: str, out: str | None, with_archives: bool, with_milestones: bool,
           allow_live: bool = False) -> int:
    import torch
    from fishrl.train import ownership
    from fishrl.train.checkpoint import latest_path, load_checkpoint
    from fishrl.train.locks import is_locked

    latest = latest_path(ckpt_dir)
    if not os.path.exists(latest):
        print(f"[export] no checkpoint at {latest}; nothing to export", flush=True)
        return 1
    # Relay discipline: a live trainer means latest.pt is a moving target and the
    # ownership release below would strand a still-running owner. trainer.lock is
    # held for the trainer's whole lifetime, so a held lock == running trainer.
    if is_locked(ownership.trainer_lock_path(ckpt_dir)):
        if not allow_live:
            print(f"[export] a trainer is LIVE on {ckpt_dir} (trainer.lock held); "
                  f"stop it first for a clean relay handoff. --allow-live overrides "
                  f"(snapshot copy only: ownership will NOT be released)", flush=True)
            return 1
        print("[export] WARNING: exporting under a live trainer (--allow-live); this "
              "zip is a snapshot copy, NOT a relay handoff -- ownership stays here and "
              "importing it elsewhere forks the lineage", flush=True)
    pl = load_checkpoint(latest, map_location="cpu")   # doubles as the sanity check
    done = int(pl.get("done", 0))

    names = _members(ckpt_dir, with_archives, with_milestones)
    for n in _CORE:
        if n not in names:
            print(f"[export] note: {n} absent, continuing without it", flush=True)

    # Relay handoff bookkeeping: a normal export RELEASES local ownership (after the
    # zip is safely cut) and the zip carries the post-release generation, so the
    # importer's claim lines up. An --allow-live snapshot copy releases nothing and
    # carries the current generation.
    handoff = not allow_live
    prev_stamp = ownership.read(ckpt_dir)
    cur_gen = int(prev_stamp["generation"]) if prev_stamp else 0
    zip_gen = cur_gen + 1 if handoff else cur_gen

    manifest = {
        "schema": SCHEMA,
        "created_unix": time.time(),
        "created_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": _git_commit(),
        "done": done,
        "elapsed_h": float(pl.get("elapsed", 0.0)) / 3600.0,
        "config": pl.get("config", {}),
        "files": {n: {"size": os.path.getsize(os.path.join(ckpt_dir, n)),
                      "sha256": _sha256(os.path.join(ckpt_dir, n))} for n in names},
        "with_archives": with_archives,
        "with_milestones": with_milestones,
        "owner_host": ownership.this_host(),
        "generation": zip_gen,
        "handoff": handoff,
    }

    if out is None:
        out = f"fishrl_run_{done:08d}_{time.strftime('%Y%m%d-%H%M')}.zip"
    tmp = out + ".tmp"
    with zipfile.ZipFile(tmp, "w") as z:
        for n in names:                                # .pt files are already zip containers
            comp = zipfile.ZIP_STORED if n.endswith(".pt") else zipfile.ZIP_DEFLATED
            z.write(os.path.join(ckpt_dir, n), n, compress_type=comp)
        z.writestr(MANIFEST, json.dumps(manifest, indent=2),
                   compress_type=zipfile.ZIP_DEFLATED)
    os.replace(tmp, out)
    size_mb = os.path.getsize(out) / 1e6
    print(f"[export] it {done} -> {out} ({size_mb:.1f} MB, {len(names)} files)", flush=True)
    if handoff:
        ownership.release(ckpt_dir)                    # zip is safe on disk; hand the turn over
        print(f"[export] ownership released (generation {zip_gen}); this dir will refuse "
              f"to train until the lineage is imported back or explicitly claimed",
              flush=True)
    return 0


def import_run(zip_path: str, ckpt_dir: str, force: bool, merge_stats: bool,
               force_stale: bool = False) -> int:
    import torch
    from fishrl.train import ownership
    from fishrl.train import stats as stats_io
    from fishrl.train.checkpoint import latest_path, load_checkpoint

    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        if MANIFEST not in names or "latest.pt" not in names:
            print(f"[import] {zip_path} is not a fishrl run export "
                  f"(missing {MANIFEST} or latest.pt)", flush=True)
            return 1
        manifest = json.loads(z.read(MANIFEST))

        local_minor = ".".join(torch.__version__.split(".")[:2])
        source_minor = ".".join(str(manifest.get("torch", "")).split(".")[:2])
        if source_minor and source_minor != local_minor:
            print(f"[import] WARNING: export was saved by torch {manifest['torch']}, "
                  f"local is {torch.__version__} — loading newer-into-older can fail; "
                  f"keep both machines on the requirements.txt pin", flush=True)

        # Stale-zip guard: if this dir's lineage has already handed off PAST the
        # zip's generation, importing it would silently rewind the relay. The
        # local stamp only ever exceeds the zip after a newer export happened
        # here, so equal generations (the normal round-trip) always pass.
        local_stamp = ownership.read(ckpt_dir)
        zip_gen = manifest.get("generation")
        if (local_stamp is not None and zip_gen is not None
                and int(local_stamp.get("generation", 0)) > int(zip_gen)):
            if not force_stale:
                print(f"[import] STALE zip: it carries generation {zip_gen} but "
                      f"{ckpt_dir} is already at generation "
                      f"{local_stamp['generation']} -- the lineage moved on past this "
                      f"export. Pass --force-stale (with --force) only if you really "
                      f"mean to rewind to it.", flush=True)
                return 1
            print("[import] WARNING: importing a stale-generation zip (--force-stale); "
                  "this rewinds the relay lineage to an older export", flush=True)

        latest = latest_path(ckpt_dir)
        if os.path.exists(latest) and not force:
            print(f"[import] {latest} exists; pass --force to overwrite it", flush=True)
            return 1

        tmp_dir = os.path.join(ckpt_dir, ".import_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        payload_names = [n for n in names if n != MANIFEST]
        z.extractall(tmp_dir, members=payload_names)

    got = _sha256(os.path.join(tmp_dir, "latest.pt"))
    want = manifest.get("files", {}).get("latest.pt", {}).get("sha256")
    if want and got != want:
        print("[import] latest.pt sha256 mismatch vs manifest; aborting", flush=True)
        return 1
    pl = load_checkpoint(os.path.join(tmp_dir, "latest.pt"), map_location="cpu")
    missing = {"format", "config", "models", "frozen", "done"} - set(pl)
    if missing:
        print(f"[import] latest.pt missing payload keys {sorted(missing)}; aborting", flush=True)
        return 1

    incoming_stats = None
    for n in payload_names:
        src = os.path.join(tmp_dir, n)
        if n == "stats.json" and merge_stats and os.path.exists(stats_io.stats_path(ckpt_dir)):
            with open(src) as f:
                incoming_stats = json.load(f)          # merged below, not replaced
            os.remove(src)
            continue
        os.replace(src, os.path.join(ckpt_dir, n))
    if incoming_stats is not None:
        added = stats_io.merge(ckpt_dir, reports=incoming_stats.get("reports"),
                               evals=incoming_stats.get("evals"))
        print(f"[import] merged stats histories: {added}", flush=True)
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    # Take the relay turn: this host is now the active owner at the zip's
    # generation (old zips without one keep whatever generation was local).
    stamp = ownership.claim(ckpt_dir, generation=zip_gen)
    print(f"[import] it {int(pl.get('done', 0))} -> {ckpt_dir} "
          f"(owner: {stamp['host']}, generation {stamp['generation']})", flush=True)
    print(f"[import] resume with: python -m fishrl.train --resume --iters 0 "
          f"--ckpt-dir {ckpt_dir}", flush=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Export/import a fishrl training run.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("export", help="package a run's checkpoints into a zip")
    ex.add_argument("--ckpt-dir", default="checkpoints")
    ex.add_argument("--out", default=None, help="zip path (default: fishrl_run_<it>_<ts>.zip)")
    ex.add_argument("--with-archives", action="store_true",
                    help="include the permanent archive_*.pt history (can be large)")
    ex.add_argument("--with-milestones", action="store_true",
                    help="include the rolling step_*.pt milestones")
    ex.add_argument("--allow-live", action="store_true",
                    help="export even while a trainer is running: a snapshot COPY, not "
                         "a relay handoff (ownership is not released)")

    im = sub.add_parser("import", help="unpack a run export into a checkpoint dir")
    im.add_argument("--zip", required=True, dest="zip_path")
    im.add_argument("--ckpt-dir", default="checkpoints")
    im.add_argument("--force", action="store_true",
                    help="overwrite an existing latest.pt in --ckpt-dir")
    im.add_argument("--force-stale", action="store_true",
                    help="import a zip whose relay generation is OLDER than this dir's "
                         "(rewinds the lineage; almost never what you want)")
    im.add_argument("--merge-stats", action="store_true",
                    help="union the zip's stats.json into the existing one "
                         "(existing rows win) instead of replacing/skipping it")

    cl = sub.add_parser("claim", help="take relay ownership of a checkpoint dir "
                                      "(recovery for an interrupted handoff, or a "
                                      "deliberate fork)")
    cl.add_argument("--ckpt-dir", default="checkpoints")
    args = ap.parse_args()

    if args.cmd == "export":
        sys.exit(export(args.ckpt_dir, args.out, args.with_archives, args.with_milestones,
                        allow_live=args.allow_live))
    if args.cmd == "claim":
        from fishrl.train import ownership
        prev = ownership.read(args.ckpt_dir)
        stamp = ownership.claim(args.ckpt_dir)
        was = (f"was {prev['state']} by '{prev['host']}' at generation "
               f"{prev['generation']}" if prev else "was unstamped")
        print(f"[claim] {args.ckpt_dir}: now active for '{stamp['host']}' at "
              f"generation {stamp['generation']} ({was})", flush=True)
        sys.exit(0)
    sys.exit(import_run(args.zip_path, args.ckpt_dir, args.force, args.merge_stats,
                        force_stale=args.force_stale))


if __name__ == "__main__":
    main()
