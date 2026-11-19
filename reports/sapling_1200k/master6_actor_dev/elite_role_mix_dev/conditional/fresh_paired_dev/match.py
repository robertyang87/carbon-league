"""Paired-arm, two-seat fresh development trial using a frozen original examiner."""
import fcntl
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE=Path(__file__).resolve().parent
C=HERE.parent
A=C.parents[1]
sys.path.insert(0,str(A/"first_win_dev"))
import native

native.HERE=HERE
native.PLAN=HERE/"protocol.json"
native.OUT=HERE/"games"
native.REG=native.sha(native.PLAN)
p=native.read(native.PLAN)
orig_driver=native.driver

def bound_driver():
    d=orig_driver()
    d.PARENT=A/"actor_partner_dev/candidate.py"
    d.CANDIDATE=C/"candidate.py"
    return d

native.driver=bound_driver


def play(job):
    native.clock.run_one(bound_driver(),job)
    return dict(arm=job[0],**native.check(job))


def run():
    second=native.read(C/"response_gate.json")["results"][1]
    first=native.read(C/"paired_warm_replication/result.json")
    assert p["seeds"]==[621758335] and p["seats"]==[0,1] and p["eligible_prior_inputs"]==[
        native.sha(C/"paired_warm_replication/result.json"),native.sha(C/"response_gate.json")]
    assert all(r["matched_initial_rng_and_cost"] and r["delta"]>0 and r["response_changed_turns"] for r in (first,second))
    d=bound_driver()
    assert native.sha(d.CANDIDATE)==p["candidate"] and native.sha(d.PARENT)==p["parent"] and native.sha(d.arena.prior.SOURCE)==p["original"]
    s=p["seeds"][0]
    with native.SHARED.open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        terrain=native.clearance()["physical_boards"][str(s)]
        for ext in [C.parent/"protocol.json",C/"protocol.json"]:
            assert all(r["terrain_sha256"]!=terrain and r["seed"]!=s for r in native.read(ext)["external_games"])
        native.OUT.mkdir(exist_ok=True)
        binding=dict(protocol=native.REG,candidate=p["candidate"],parent=p["parent"],original=p["original"],terrain=terrain)
        rows=[]
        for pair in [[("candidate",0),("parent",1)],[("parent",0),("candidate",1)]]:
            jobs=[(arm,s,seat,seat,terrain,binding) for arm,seat in pair]
            with ProcessPoolExecutor(2) as pool:
                for r in pool.map(play,jobs):
                    rows.append(r);print(r["arm"],r["seat"],r["score_by_seat"],r["margin"],flush=True)
                    native.save(HERE/"progress.json",rows)
        native.save(HERE/"scores.json",rows)


if __name__=="__main__":run()
