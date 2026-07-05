"""One-command relay passovers: train the ODROID's lineage on this machine.

    python -m fishrl.relay train                  # full cycle: pull -> train -> hand back
    python -m fishrl.relay train -- --reserve-cores 2 --collect-workers 8   # trainer flags
    python -m fishrl.relay pull                   # just take the turn (no training)
    python -m fishrl.relay handback               # just send the lineage home
    python -m fishrl.relay status                 # both sides' ownership + service state

``train`` automates the whole handoff the runbook describes by hand:

  1. stop the remote systemd trainer (graceful checkpoint; ``ssh -t`` so sudo
     may prompt -- add a NOPASSWD sudoers line for the two systemctl commands
     to make it prompt-free),
  2. ``transfer export`` on the server (releases ITS ownership), fetch the zip,
  3. ``transfer import --force --merge-stats`` here (claims ownership),
  4. run the trainer in the foreground -- Ctrl-C is the normal way to end the
     session (the trainer checkpoints gracefully; this wrapper then ignores
     further Ctrl-C while it hands the result back),
  5. export here, push the zip, import on the server, restart the unit.

Every step rides the ownership protocol (fishrl.train.ownership), so an
interruption anywhere leaves at most one owner and a clear next move:
re-run ``train`` (it skips the pull leg when this machine already owns the
lineage), run ``handback``, or ``python -m fishrl.transfer claim`` on
whichever machine should resume. Worst case is never corruption.

If this machine already owns the lineage (a previous session that never
handed back), ``train`` skips the pull leg automatically.

``--local-remote DIR`` swaps the ssh/scp backend for plain local
subprocesses against another checkpoint dir on this machine -- the
simulation backend the tests (and a cautious first run) use; it implies
``--no-service``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

# The production server layout (deploy/README.md); every value has a flag.
DEFAULT_HOST = "odroid-lan"
DEFAULT_REMOTE_REPO = "/home/zubekanov/Repositories/How-To-Train-Your-Fish"
DEFAULT_REMOTE_PYTHON = "/home/zubekanov/Repositories/Website_Dev/.venv/bin/python"
DEFAULT_UNIT = "fishrl-selfplay"
REMOTE_ZIP = "/tmp/fishrl_relay.zip"                   # server-side scratch, both directions


class Ssh:
    """The training server, over ssh/scp (OpenSSH; Windows 10 ships a client).
    ``tty=True`` allocates a terminal so ``sudo`` can prompt for a password."""

    def __init__(self, host: str, repo: str, python: str, unit: str, no_service: bool):
        self.host, self.repo, self.python = host, repo, python
        self.unit, self.no_service = unit, no_service

    def _ssh(self, cmd: str, tty: bool = False, check: bool = True):
        args = (["ssh", "-o", "ConnectTimeout=10"] + (["-t"] if tty else [])
                + [self.host, cmd])
        return subprocess.run(args, check=check)

    def preflight(self) -> None:
        print(f"[relay] preflight: ssh {self.host} ...", flush=True)
        r = self._ssh("echo relay-ok", check=False)
        if r.returncode != 0:
            raise SystemExit(f"[relay] cannot reach '{self.host}' over ssh; aborting "
                             f"before touching anything")
        r = self._ssh(f"cd {self.repo} && {self.python} -c "
                      f"'import fishrl.train.ownership'", check=False)
        if r.returncode != 0:
            raise SystemExit(f"[relay] the checkout on {self.host} predates the relay "
                             f"protocol -- git pull there first")

    def stop_service(self) -> None:
        if self.no_service:
            return
        print(f"[relay] stopping {self.unit} on {self.host} (graceful checkpoint; "
              f"sudo may prompt)", flush=True)
        self._ssh(f"sudo systemctl stop {self.unit}", tty=True)

    def start_service(self) -> None:
        if self.no_service:
            return
        print(f"[relay] starting {self.unit} on {self.host}", flush=True)
        self._ssh(f"sudo systemctl start {self.unit}", tty=True)
        r = self._ssh(f"systemctl is-active {self.unit}", check=False)
        if r.returncode != 0:
            raise SystemExit(f"[relay] {self.unit} did not come back up -- check "
                             f"'journalctl -u {self.unit}' on {self.host}")

    def service_state(self) -> str:
        if self.no_service:
            return "n/a"
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=10", self.host,
                            f"systemctl is-active {self.unit}"],
                           capture_output=True, text=True)
        return (r.stdout or r.stderr).strip() or "unknown"

    def _transfer(self, argline: str) -> None:
        self._ssh(f"cd {self.repo} && {self.python} -m fishrl.transfer {argline}")

    def export(self) -> None:
        self._transfer(f"export --ckpt-dir checkpoints --out {REMOTE_ZIP}")

    def import_back(self) -> None:
        self._transfer(f"import --zip {REMOTE_ZIP} --ckpt-dir checkpoints "
                       f"--force --merge-stats")

    def owner(self) -> str:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=10", self.host,
                            f"cat {self.repo}/checkpoints/owner.json"],
                           capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else "(no owner.json)"

    def fetch(self, local_zip: str) -> None:
        subprocess.run(["scp", "-q", f"{self.host}:{REMOTE_ZIP}", local_zip], check=True)

    def send(self, local_zip: str) -> None:
        subprocess.run(["scp", "-q", local_zip, f"{self.host}:{REMOTE_ZIP}"], check=True)

    def cleanup(self) -> None:
        self._ssh(f"rm -f {REMOTE_ZIP}", check=False)


class LocalSim(Ssh):
    """Simulation backend: the 'server' is another checkpoint dir on THIS
    machine (tests / dry runs). Transfer runs in-process; there is no service."""

    def __init__(self, ckpt_dir: str):
        super().__init__("local-sim", "", sys.executable, "(none)", no_service=True)
        self.ckpt = ckpt_dir
        self.zip = os.path.join(tempfile.gettempdir(), "fishrl_relay_sim.zip")

    def preflight(self) -> None:
        if not os.path.exists(os.path.join(self.ckpt, "latest.pt")):
            raise SystemExit(f"[relay] no latest.pt in --local-remote {self.ckpt}")

    def service_state(self) -> str:
        return "n/a"

    def export(self) -> None:
        from fishrl.transfer import export
        if export(self.ckpt, self.zip, False, False) != 0:
            raise SystemExit("[relay] export on the (simulated) server failed")

    def import_back(self) -> None:
        from fishrl.transfer import import_run
        if import_run(self.zip, self.ckpt, force=True, merge_stats=True) != 0:
            raise SystemExit("[relay] import on the (simulated) server failed")

    def owner(self) -> str:
        try:
            with open(os.path.join(self.ckpt, "owner.json")) as f:
                return f.read().strip()
        except OSError:
            return "(no owner.json)"

    def fetch(self, local_zip: str) -> None:
        shutil.copy2(self.zip, local_zip)

    def send(self, local_zip: str) -> None:
        shutil.copy2(local_zip, self.zip)

    def cleanup(self) -> None:
        for p in (self.zip,):
            try:
                os.remove(p)
            except OSError:
                pass


# ── the three legs ───────────────────────────────────────────────────────────

def _local_owner_active(ckpt_dir: str) -> bool:
    from fishrl.train import ownership
    stamp = ownership.read(ckpt_dir)
    return (stamp is not None and stamp.get("state") == "active"
            and stamp.get("host") == ownership.this_host())


def pull(remote: Ssh, ckpt_dir: str) -> None:
    """Take the turn: stop the server's trainer, export there, import here."""
    from fishrl.transfer import import_run
    remote.preflight()
    remote.stop_service()
    remote.export()                                    # releases the server's ownership
    local_zip = os.path.join(tempfile.gettempdir(), "fishrl_relay_pull.zip")
    remote.fetch(local_zip)
    try:
        if import_run(local_zip, ckpt_dir, force=True, merge_stats=True) != 0:
            raise SystemExit(
                "[relay] local import failed -- the server keeps its released stamp; "
                "fix the issue and re-run, or 'python -m fishrl.transfer claim' on "
                "the server to resume training there instead")
    finally:
        try:
            os.remove(local_zip)
        except OSError:
            pass
    remote.cleanup()
    print("[relay] pull complete: this machine owns the lineage", flush=True)


def handback(remote: Ssh, ckpt_dir: str) -> None:
    """Send the lineage home: export here, import + restart on the server."""
    from fishrl.transfer import export
    remote.preflight()
    local_zip = os.path.join(tempfile.gettempdir(), "fishrl_relay_back.zip")
    if export(ckpt_dir, local_zip, False, False) != 0:  # refuses under a live trainer
        raise SystemExit("[relay] local export failed (trainer still running?); "
                         "nothing was sent")
    try:
        remote.send(local_zip)
        remote.import_back()                            # server claims the turn
    finally:
        try:
            os.remove(local_zip)
        except OSError:
            pass
    remote.start_service()
    remote.cleanup()
    print("[relay] handback complete: the server owns the lineage again", flush=True)


def run_trainer(ckpt_dir: str, extra: list) -> int:
    """Foreground trainer session. Ctrl-C lands on the CHILD too (same console
    group), which is exactly the trainer's graceful-checkpoint path -- here we
    just keep waiting until it exits, however many times Ctrl-C is pressed."""
    cmd = [sys.executable, "-m", "fishrl.train", "--resume", "--iters", "0",
           "--ckpt-dir", ckpt_dir] + list(extra)
    print(f"[relay] training: {' '.join(cmd)}", flush=True)
    print("[relay] Ctrl-C ends the session: the trainer checkpoints, then the "
          "result is handed back automatically", flush=True)
    proc = subprocess.Popen(cmd)
    while True:
        try:
            rc = proc.wait()
            break
        except KeyboardInterrupt:
            continue                                    # child is checkpointing; wait it out
    return rc


def train_cycle(remote: Ssh, ckpt_dir: str, extra: list, no_return: bool) -> None:
    if _local_owner_active(ckpt_dir):
        print(f"[relay] {ckpt_dir} is already this machine's turn -- skipping the "
              f"pull leg (a previous session that never handed back)", flush=True)
    else:
        pull(remote, ckpt_dir)
    rc = run_trainer(ckpt_dir, extra)
    if rc != 0:
        print(f"[relay] trainer exited with code {rc}; handing back anyway (the "
              f"last checkpoint is the lineage state)", flush=True)
    if no_return:
        print("[relay] --no-return: lineage stays here; run "
              "'python -m fishrl.relay handback' when done", flush=True)
        return
    # The session is over -- don't let an extra Ctrl-C kill the handback and
    # strand the lineage mid-flight.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for attempt in range(3):
        try:
            handback(remote, ckpt_dir)
            return
        except (subprocess.CalledProcessError, SystemExit) as e:
            if attempt == 2:
                print(f"[relay] handback failed ({e}); the lineage is SAFE here -- "
                      f"re-run 'python -m fishrl.relay handback' when the server is "
                      f"reachable", flush=True)
                raise
            print(f"[relay] handback attempt {attempt + 1} failed ({e}); retrying "
                  f"in 10s", flush=True)
            time.sleep(10)


def status(remote: Ssh, ckpt_dir: str) -> None:
    from fishrl.train import ownership
    from fishrl.train.locks import is_locked
    stamp = ownership.read(ckpt_dir)
    held = is_locked(ownership.trainer_lock_path(ckpt_dir))
    print(f"[relay] local  {ckpt_dir}: owner={stamp or '(none)'} "
          f"trainer={'RUNNING' if held else 'stopped'}", flush=True)
    remote.preflight()
    print(f"[relay] remote {remote.host}: unit={remote.service_state()} "
          f"owner={remote.owner()}", flush=True)


def parse(argv: list) -> tuple:
    """(args, trainer_extra) from an argv WITHOUT the program name. The trainer
    passthrough is split off at the literal ``--`` BEFORE argparse sees anything."""
    ap = argparse.ArgumentParser(
        prog="fishrl.relay",
        description="Automated relay passovers between this machine and the training server.")
    ap.add_argument("cmd", choices=["train", "pull", "handback", "status"])
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    ap.add_argument("--remote-python", default=DEFAULT_REMOTE_PYTHON)
    ap.add_argument("--unit", default=DEFAULT_UNIT)
    ap.add_argument("--no-service", action="store_true",
                    help="skip the systemctl stop/start (server trainer managed by hand)")
    ap.add_argument("--no-return", action="store_true",
                    help="train only; keep the lineage here (handback later)")
    ap.add_argument("--local-remote", default=None, metavar="DIR",
                    help="simulate the server with a local checkpoint dir (testing)")
    ap.epilog = ("args after a literal -- go to fishrl.train "
                 "(e.g. -- --reserve-cores 2 --collect-workers 8)")

    # Split the trainer passthrough BEFORE argparse. argparse.REMAINDER is not
    # usable here: it greedily swallows every token after the subcommand --
    # including this tool's own --ckpt-dir/--local-remote flags -- which once
    # sent a "simulation" run to the real server on pure defaults.
    extra: list = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]
    return ap.parse_args(argv), extra


def main() -> None:
    args, extra = parse(sys.argv[1:])

    if args.cmd in ("train", "pull", "handback"):
        # The relay must outlast the trainer awake: the handback export/scp runs
        # AFTER the trainer exits, exactly when an idle-sleep timeout would
        # otherwise put the PC to sleep mid-transfer.
        from fishrl.train.keepawake import keep_awake
        keep_awake(f"relay {args.cmd}")

    if args.local_remote is not None:
        remote: Ssh = LocalSim(args.local_remote)
        print(f"[relay] target: LOCAL SIMULATION dir {args.local_remote}", flush=True)
    else:
        remote = Ssh(args.host, args.remote_repo, args.remote_python, args.unit,
                     args.no_service)
        print(f"[relay] target: {args.host} repo={args.remote_repo} unit={args.unit} | "
              f"local ckpt-dir: {args.ckpt_dir}", flush=True)

    if args.cmd == "train":
        train_cycle(remote, args.ckpt_dir, extra, args.no_return)
    elif args.cmd == "pull":
        pull(remote, args.ckpt_dir)
    elif args.cmd == "handback":
        handback(remote, args.ckpt_dir)
    else:
        status(remote, args.ckpt_dir)


if __name__ == "__main__":
    main()
