"""Verify every archived action and measure whether each live pressure switch reached a loaded opponent."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import tree_audited_game as arena
from turning_point_ledger import CashEvents
from pressure_return_features import _PRESSURE_DEST

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/"all_top5_pressure_dev"
MOVES={None:0,"NONE":0,"UP":1,"RIGHT":2,"DOWN":3,"LEFT":4}


def check(game):
    path=OUT/"candidate"/game["raw_file"];payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=game["raw_sha256"]:raise ValueError("raw identity differs")
    raw=json.loads(gzip.decompress(payload));seat=raw["candidate_seat"]
    candidate=ROOT/"data/weights/sapling_1200k/pressure_return_it50_v3.py"
    parent=ROOT/"data/weights/sapling_1200k/hybrid_it50_v3.py"
    if arena.prior.digest(candidate)!=raw["pack_sha256"] or arena.prior.digest(parent)!="b54cc51f4109f19f4a7b508eafc9cf04a28b7e295a9f442c5208f0d33f10a9ca":
        raise ValueError("wrong actor")
    live=arena.prior.shared.module(candidate,"live_pressure")
    no_student=arena.prior.shared.module(candidate,"pressure_without_student");no_student._HYBRID["student_weight"]=0.
    actor=arena.prior.shared.module(parent,"original_hybrid")
    sim=arena.exam.FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"]);sim.cash=CashEvents(sim.cash)
    counts=Counter();changes=[]
    for t,orders in enumerate(raw["actions"]):
        obs=sim.obs(seat);a=live.agent(obs,sim.config);bare=actor.agent(obs,sim.config);nost=no_student.agent(obs,sim.config)
        if a!=orders[seat]:raise ValueError(f"frozen actor mismatch {seat=} {t=}")
        student_diffs={k for k in a.keys()|nost.keys() if a.get(k)!=nost.get(k)}
        counts["roots"]+=1;counts["sapling_changed_roots"]+=bool(student_diffs);counts["sapling_changed_orders"]+=len(student_diffs)
        changed={k for k in a.keys()|bare.keys() if a.get(k)!=bare.get(k)}
        if bool(live._PRESSURE_LAST)!=bool(changed):raise ValueError("pressure counter disagrees with actual joint order")
        if changed:
            item=dict(live._PRESSURE_LAST);uid=item["hunter"];v=item["victim"]
            if changed!={uid}:raise ValueError("intervention affected unrelated unit")
            h=sim.workers[uid];victim=sim.workers[v]
            ours=int(_PRESSURE_DEST[h.pos,MOVES[a.get(uid)]])
            theirs=int(_PRESSURE_DEST[victim.pos,MOVES[orders[1-seat].get(v)]])
            item.update(step=t,actual=a.get(uid),parent=bare.get(uid),victim_action=orders[1-seat].get(v),
                        our_destination=ours,victim_destination=theirs,realized_crossing=(ours==theirs))
        sim.cash.events=[]
        sim.step(*orders)
        if changed:
            row=next((r for r in sim.last_collisions if r[0]==v),None)
            item["victim_died"]=v not in sim.workers
            item["hunter_survived"]=uid in sim.workers
            item["observed_victim_cargo_to_our_hunter"]=float(row[2]) if (
                item["realized_crossing"] and uid in sim.workers and row and row[3]==seat and row[4]==seat
            ) else 0.
            item["observed_victim_bank_this_turn"]=sum(e[2] for e in sim.cash.events if e[0]=="own_delivery" and e[1]==1-seat and e[3]==v)
            changes.append(item)
    if arena.prior.hashed(sim.get_state())!=game["final_state_sha256"]:raise ValueError("terminal replay mismatch")
    return {"seed":raw["seed"],"seat":seat,"raw_sha256":game["raw_sha256"],"counts":dict(counts),
            "pressure":dict(live._PRESSURE_COUNTS),"slot_attribution":live._COUNTS,"changes":changes}


if __name__=="__main__":
    report=json.loads((OUT/"candidate/results.json").read_bytes())
    if report["status"]!="complete" or {(g["seed"],g["candidate_seat"]) for g in report["games"]}!={(621455920,0),(621455920,1)}:
        raise ValueError("incomplete diagnostic")
    rows=[check(g) for g in report["games"]]
    (OUT/"candidate/attribution.json").write_text(json.dumps({"games":rows},ensure_ascii=False,indent=2)+"\n")
