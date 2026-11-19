"""Registered new-map check; only starts after both independent return comparisons succeed."""
from concurrent.futures import ProcessPoolExecutor
import fcntl
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parents[1]
sys.path.insert(0,str(ACTOR/"first_win_dev"))
import native
native.HERE=HERE; native.PLAN=HERE/"protocol.json"; native.OUT=HERE/"games";native.REG=native.sha(native.PLAN)
plan=native.read(native.PLAN)
base_driver=native.driver


def bound():
    d=base_driver();d.PARENT=ACTOR/"actor_partner_dev/candidate.py";d.CANDIDATE=HERE/"candidate.py"
    return d


native.driver=bound


def play(job):
    native.clock.run_one(bound(),job)
    return dict(arm=job[0],**native.check(job))


def main():
    gate=native.read(HERE/"gate.json")
    d=bound()
    if not gate["eligible"] or gate["protocol"]!=native.REG:
        raise SystemExit("Development gate closed")
    assert all(z["delta"]>0 and z["changed_enemy_steps"] for z in gate["rows"]) and len(gate["rows"])==2
    assert native.sha(d.PARENT)==plan["parent"] and native.sha(d.CANDIDATE)==plan["candidate"] and native.sha(d.arena.prior.SOURCE)==plan["original"]
    s=plan["seeds"][0]
    with native.SHARED.open("a") as slot:
        fcntl.flock(slot,fcntl.LOCK_EX)
        terrain=native.clearance()["physical_boards"][str(s)]
        panels=[HERE.parent/"protocol.json",HERE.parent/"conditional/protocol.json",HERE.parent/"recruit_only_dev/protocol.json"]
        for f in panels:
            x=native.read(f); rs=x.get("external_games",x.get("games",[]))
            assert all(g.get("seed")!=s and g.get("terrain_sha256")!=terrain for g in rs)
        old=native.read(HERE.parent/"conditional/fresh_paired_dev/clearance.json")["physical_boards"];assert str(s) not in old and terrain not in old.values()
        native.OUT.mkdir(exist_ok=True)
        auth=dict(protocol=native.REG,candidate=plan["candidate"],parent=plan["parent"],original=plan["original"],terrain=terrain)
        rows=[]
        for wave in [[("candidate",0),("parent",1)],[("parent",0),("candidate",1)]]:
            jobs=[(arm,s,seat,seat,terrain,auth) for arm,seat in wave]
            with ProcessPoolExecutor(2) as pool:
                for row in pool.map(play,jobs):
                    rows.append(row)
                    print(row["arm"],row["seat"],row["score_by_seat"],row["margin"],flush=True)
        native.save(HERE/"scores.json",rows)


if __name__=="__main__":main()
