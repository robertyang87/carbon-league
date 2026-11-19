"""Run one new paired development board only after both observed-response receipt gates pass."""
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from fit import HERE, ROOT, load, save
from package import build, verify

ACTOR=HERE.parent
sys.path.insert(0,str(ACTOR/"first_win_dev"))
import native

PLAN=HERE/"match_plan.json"
CAND=HERE/"candidate.py"
PARENT=ACTOR/"planter_capture_full_dev/candidate.py"

def play(job):
    driver=native.driver()
    driver.CANDIDATE,driver.PARENT=CAND,PARENT
    native.clock.run_one(driver,job)
    return native.check(job)

def main():
    audited=load(HERE/"native_settlement.json")
    if not audited["passes_receipt_and_margin_gate"]:raise ValueError("native receipt gate failed")
    build(CAND);verify(CAND)
    original=native.sha(ROOT/"data/master3s2_source.py")
    parity=load(HERE/"package_parity.json")
    plan=dict(seeds=[621758332],seats=[0,1],arms=["candidate","parent"],formal=False,
              source_revision=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
              original=original,candidate=native.sha(CAND),parent=native.sha(PARENT),
              native_receipt_sha256=native.sha(HERE/"native_settlement.json"),gate_sha256=native.sha(HERE/"protocol.json"))
    if plan["candidate"]!=parity["candidate_sha"] or plan["parent"]!=parity["parent_sha"]:raise ValueError("fresh candidate changed")
    if PLAN.exists():
        if load(PLAN)!=plan:raise ValueError("fresh run binding changed")
    else:save(PLAN,plan)
    native.HERE=HERE;native.PLAN=PLAN;native.OUT=HERE/"games";native.REG=native.sha(PLAN)
    with native.SHARED.open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        terrain=native.clearance()["physical_boards"]["621758332"]
        bind=dict(protocol=native.REG,candidate=plan["candidate"],parent=plan["parent"],original=original,terrain=terrain)
        rows=[]
        for arm in ("candidate","parent"):
            jobs=[(arm,621758332,s,s if arm=="candidate" else 1-s,terrain,bind) for s in (0,1)]
            with ProcessPoolExecutor(2) as workers:
                rows.extend(dict(arm=arm,**r) for r in workers.map(play,jobs))
        save(HERE/"scores.json",rows)

if __name__=="__main__":main()
