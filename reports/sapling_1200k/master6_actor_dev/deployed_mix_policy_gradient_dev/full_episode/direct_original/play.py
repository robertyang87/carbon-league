"""Run the registered direct matches with the original examiner and signed clocks."""
import fcntl
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FIT = HERE.parent
ACTOR = FIT.parents[1]
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
    d.CANDIDATE = FIT / "selected.py"
    d.PARENT = native.ROOT / native.read(FIT / "protocol.json")["parent"]["path"]
    assert native.sha(d.CANDIDATE) == plan["candidate"]
    assert native.sha(d.PARENT) == plan["parent"]
    assert native.sha(d.arena.prior.SOURCE) == plan["original"]
    return d

native.driver = driver

def play(job):
    native.clock.run_one(driver(), job)
    row = dict(arm=job[0], **native.check(job))
    native.save(HERE / f"complete_{job[0]}_{job[2]}.json", row)
    return row

def main():
    seed, = plan["seeds"]
    clearance = native.clearance()
    board = clearance["physical_boards"][str(seed)]
    for reg in ACTOR.rglob("clearance.json"):
        content = native.read(reg)
        others = content.get("physical_boards", {}) if isinstance(content, dict) else {}
        if reg.parent != HERE and isinstance(others, dict):
            assert str(seed) not in others and board not in others.values(), reg
    assert board not in native.read(FIT / "register.json")["boards"].values()
    for game in native.read(ACTOR / "broader_elite_backbone_dev/panel.json")["games"]:
        assert game["terrain_sha256"] != board and game["seed"] != seed
    driver()
    binding = {k:plan[k] for k in ("original","candidate","parent")}
    binding.update(protocol=native.REG, program=native.sha(Path(__file__)), terrain=board)
    native.OUT.mkdir(exist_ok=True)
    rows = []
    with native.SHARED.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for wave in plan["schedule"]:
            jobs = [(arm, seed, seat, seat, board, binding) for arm,seat in wave]
            with ProcessPoolExecutor(2) as pool:
                for row in pool.map(play, jobs):
                    rows.append(row)
                    print(row["arm"], row["seat"], row["score_by_seat"], flush=True)
    native.save(HERE / "scores.json", rows)

if __name__ == "__main__":
    main()
