"""Verify each official balance category and the cargo recipient against a separate simulator."""
from collections import Counter
from pathlib import Path
import sys

import analyze as study
sys.path.insert(0,str(study.BASE))
from turning_point_ledger import CashEvents, FastCarbon


def game(g,tape):
    sim=FastCarbon(tape["game_configuration"]).reset(seed=g["seed"])
    sim.cash=CashEvents(sim.cash)
    checks=Counter()
    for turn,joint in zip(g["turns"],tape["actions"],strict=True):
        sim.cash.events.clear()
        sim.step(*joint)
        channels=[Counter(),Counter()]
        for name,side,value,*_ in sim.cash.events:
            channels[side][name]+=value
        for side,c in enumerate(turn["cash_gain_by_seat"]):
            for k in set(c)|set(channels[side]):
                study.check(abs(c.get(k,0)-channels[side][k])<1.e-5,"separate-simulator cash channel differs")
            checks["seat_turns"]+=1
        actual={d["worker"]:d for d in turn["deaths"] if d["reason"]=="worker_collision"}
        study.check(len(actual)==len(sim.last_collisions),"different number of collisions")
        for wid,owner,cargo,recipient,winner in sim.last_collisions:
            d=actual[wid]
            s,beneficiary,_=d["transfer_to"]
            study.check(d["owner"]==owner and abs(d["cargo"]-cargo)<1.e-6 and recipient==beneficiary and
                        (s=="worker" and winner==beneficiary or s!="worker" and winner is None),"wrong custody")
            checks["collision_casualties"]+=1
            if cargo>0 and recipient==1-owner:
                checks["positive_cargo_to_enemy_"+s]+=1
        checks["joint_transitions"]+=1
    study.check(list(sim.cash)==g["closing_cash"],"closing balance differs")
    return {"raw_sha256":g["raw_sha256"],"verified":dict(checks)}


def corroborate(evidence):
    rows={}
    for arm,games in evidence.items():
        rows[arm]=[]
        for g in games:
            p=study.DEV/arm/f"raw_{g['seed']}_{g['seat']}.json.gz"
            study.check(study.sha(p)==g["raw_sha256"],"changed tape")
            rows[arm].append(game(g,study.read(p)))
    record={"implementation_sha256":study.sha(Path(__file__)),
            "fastsim_sha256":study.sha(study.ROOT/"code/fastsim.py"),
            "cash_events_sha256":study.sha(study.BASE/"turning_point_ledger.py"),"games":rows}
    study.publish(study.HERE/"crosscheck.json",record)
    return record
