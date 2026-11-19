"""Give original-Master3s2 jobs precedence; run the fixed transfer series on CPU 0."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from run import HERE, PLAN_SHA, plan, sha, write


def native_workers():
    active = {}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            args = [x.decode() for x in (proc / "cmdline").read_bytes().split(b"\0") if x]
            if len(args) < 2 or "python" not in Path(args[0]).name:
                continue
            for a in args[1:]:
                if not a.endswith(".py") or "opponent_transfer/" in a:
                    continue
                if ("terminal_return_fit/multimap/run.py" in a or "terminal_return_fit/replication/run.py" in a
                        or a.endswith(("tree_pack_arena.py", "terminal_return_fit/trial.py", "run_match.py"))):
                    active[proc.name] = " ".join(args[1:])
                    break
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return active


def main():
    os.sched_setaffinity(0, {0})
    with (Path(tempfile.gettempdir()) / ("transfer-" + PLAN_SHA + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        d = plan()
        previous = native_workers()
        print("native first", previous, flush=True)
        cmd = [sys.executable, str(HERE / "run.py")]
        for seed in d["maps"]:
            for seat in d["candidate_seats"]:
                for arm in ("parent", "learned"):
                    while native_workers():
                        time.sleep(8)
                    if not (HERE / "start_receipt.json").exists():
                        write(HERE / "start_receipt.json", {
                            "protocol_sha256": PLAN_SHA, "runner_sha256": sha(HERE / "run.py"),
                            "scheduler_sha256": sha(Path(__file__)), "cpu_affinity": sorted(os.sched_getaffinity(0)),
                            "prior_active_jobs": previous, "active_native_at_start": {},
                            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
                    subprocess.run([*cmd, "play", arm, str(seed), str(seat)], check=True)
        subprocess.run([*cmd, "report"], check=True)


if __name__ == "__main__":
    main()
