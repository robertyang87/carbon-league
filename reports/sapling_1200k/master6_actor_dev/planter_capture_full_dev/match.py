"""One fixed development board, both seats and both arms, full official matches."""
import fcntl
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parent
sys.path.insert(0, str(ACTOR / "first_win_dev"))
import native

native.HERE = HERE
native.PLAN = HERE / "protocol.json"
native.OUT = HERE / "games"
native.REG = native.sha(native.PLAN)
plan = native.read(native.PLAN)
CAND = HERE / "candidate.py"
PARENT = ACTOR / "planter_escape_full_dev/candidate.py"


def play(job):
    driver = native.driver()
    driver.CANDIDATE, driver.PARENT = CAND, PARENT
    native.clock.run_one(driver, job)
    return native.check(job)


if __name__ == "__main__":
    with native.SHARED.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert native.sha(CAND) == plan["candidate"] and native.sha(PARENT) == plan["parent"]
        assert native.sha(native.driver().arena.prior.SOURCE) == plan["original"]
        terrain = native.clearance()["physical_boards"][str(plan["seeds"][0])]
        binding = dict(protocol=native.REG, candidate=plan["candidate"], parent=plan["parent"], original=plan["original"], terrain=terrain)
        rows = []
        for arm in ["candidate", "parent"]:
            jobs = [(arm, plan["seeds"][0], s, s if arm == "candidate" else 1-s, terrain, binding) for s in [0, 1]]
            with ProcessPoolExecutor(2) as workers:
                for row in workers.map(play, jobs):
                    rows.append(dict(arm=arm, **row))
                    print(arm, row["seat"], row["margin"], flush=True)
        native.save(HERE / "scores.json", rows)
