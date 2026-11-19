"""Direct two-terrain comparison of the economic-return Sapling rollout policy."""
from concurrent.futures import ProcessPoolExecutor
import fcntl
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

HERE = Path(__file__).resolve().parent
FIT = HERE.parent
ACTOR = FIT.parent
FULL = ACTOR / "deployed_mix_policy_gradient_dev/full_episode"
sys.path.insert(0, str(ACTOR / "first_win_dev"))
import native

native.HERE = HERE
native.PLAN = HERE / "protocol.json"
native.REG = native.sha(native.PLAN)
native.OUT = HERE / "games"
plan = native.read(native.PLAN)
base_driver = native.driver

def driver():
    d = base_driver()
    d.PARENT = FULL / "selected.py"
    d.CANDIDATE = FIT / "candidate.py"
    assert native.sha(d.PARENT) == plan["parent"]
    assert native.sha(d.arena.prior.SOURCE) == plan["original"]
    assert native.sha(d.CANDIDATE) == native.read(HERE / "binding.json")["candidate"]
    return d

native.driver = driver

def play(job):
    native.clock.run_one(driver(),job)
    return dict(arm=job[0], **native.check(job))

def main():
    auth = native.read(HERE / "binding.json")
    assert auth["protocol"] == native.REG and auth["original"] == plan["original"]
    assert auth["candidate"] != plan["parent"] == auth["parent"]
    assert auth["candidate_weights"] == native.sha(FIT / "candidate.npz")
    assert auth["runner"] == native.sha(Path(__file__))
    assert auth["evidence"] == native.sha(FIT / auth["evidence_file"])
    assert all(auth["changed_turns_by_seat"][str(seat)] > 0 for seat in (0,1))
    assert native.sha(FIT / "contract.json") == plan["training"]
    # New ladder games can extend the registry after preregistration.
    # Preserve the original receipt and independently clear the same physical boards.
    recorded = native.read(HERE / "clearance.json")
    with TemporaryDirectory() as scratch:
        native.HERE = Path(scratch)
        try:
            current = native.clearance()
        finally:
            native.HERE = HERE
    assert current["physical_boards"] == recorded["physical_boards"]
    native.save(HERE / "runtime_clearance.json", current)
    boards = current["physical_boards"]
    held = set(boards.values())
    for reg in ACTOR.rglob("clearance.json"):
        doc = native.read(reg)
        previous = doc.get("physical_boards", {}) if isinstance(doc, dict) else {}
        if reg.parent != HERE and isinstance(previous,dict):
            assert not held.intersection(previous.values()) and not set(boards).intersection(previous)
    assert all(str(r["seed"]) not in boards and r["terrain"] not in held for r in native.read(FIT / "panel.json")["cards"])
    native.OUT.mkdir(exist_ok=True)
    rows = []
    with native.SHARED.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for i,seed in enumerate(plan["seeds"]):
            for lead in (("parent","candidate") if i else ("candidate","parent")):
                other = "parent" if lead == "candidate" else "candidate"
                jobs = [(arm, seed, seat, seat, boards[str(seed)], auth) for seat,arm in enumerate((lead,other))]
                with ProcessPoolExecutor(2) as pool:
                    for row in pool.map(play,jobs):
                        rows.append(row)
                        native.save(HERE / f"complete_{row['arm']}_{seed}_{row['seat']}.json",row)
                        print(seed,row["arm"],row["seat"],row["margin"],flush=True)
    native.save(HERE / "scores.json", rows)

if __name__ == "__main__":
    main()
