"""Recompute sample targets from public transition events and original rosters, independently of FastCarbon."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from context import Corpus
from economy_extract import visible_ledger
from future_economy import FutureEconomy, HERE, registered_indexes, open_game

# Covers each archive batch, a measured collector comeback, and an early extinction.
CASES = {
    "003aed5cd8b63c5c4d82dc67e7dd78f2-1": (0,36,100,269,289,298),
    "1441fe987d83135bd3beae549bd0ce85-1": (0,101,269,298),
    "0102093a0e31a1455aaa736c368c6207-1": (0,100,269,289,298),
    "6fe4eb3c61132e6736b0ff785d2d2878-1": (0,84,85,89,298),
    "76a7c94fee77341606d80cdfb48321eb-1": (0,71,72,85,104,105,114),
}


def check_for_side(game, t, end, physical, target, money):
    frames = game._frames[t:end+1]
    obs = [r[0]["observation"]["players"] for r in frames]
    sides = [v[physical] for v in obs]
    workers = [set(k for k,v in side[2].items() if v[2]=="COLLECTOR") for side in sides]
    conf = game._data["configuration"]
    prices = [conf["recCollectorCost"] + conf["recCollectorIncreaseCost"] * sum(
        unit[2] == "COLLECTOR" for side in all_sides for unit in side[2].values()) for all_sides in obs]
    baseline = workers[0]
    ever_at_start = any(any(v[2]=="COLLECTOR" for v in frame[0]["observation"]["players"][physical][2].values())
                        for frame in game._frames[:t+1])
    ever = ever_at_start
    present = bool(baseline)
    lost = None
    gap = None
    recovered = None
    production_gap = None
    rehire_funds = None
    fundable = None
    for i, (ids,side) in enumerate(zip(workers,sides)):
        if present and not ids and lost is None:
            lost = i
        if not ids and gap is None:
            gap = i
        ever = ever or bool(ids)
        if not ids and ever and production_gap is None:
            production_gap = i
        if production_gap is not None and ids and recovered is None:
            recovered = i
        if not ids and ever and side[0]>=prices[i] and side[1] and len(side[2])<conf["workerLimit"] and t+i<game.turns and rehire_funds is None:
            rehire_funds = i
        if not ids and side[0]>=prices[i] and side[1] and len(side[2])<conf["workerLimit"] and t+i<game.turns and fundable is None:
            fundable = i
        present = bool(ids)
    expected = {
        "own_collector_delivery": round(sum(row[physical][0] for row in money),6),
        "tree_absorption": round(sum(row[physical][2] for row in money),6),
        "collectors_start":len(baseline), "collectors_end":len(workers[-1]),
        "original_collectors_surviving":len(baseline.intersection(workers[-1])),
        "sole_starting_collector_survived":next(iter(baseline)) in workers[-1] if len(baseline)==1 else None,
        "min_collectors_including_start":min(map(len,workers)),
        "first_loss_of_last_collector_offset":lost, "first_empty_offset":gap,
        "collector_owned_by_start":ever_at_start,"first_empty_after_having_collector_offset":production_gap,
        "first_fundable_rehire_after_having_collector_offset":rehire_funds,
        "first_collector_present_again_offset":recovered, "first_fundable_hire_while_empty_offset":fundable,
        "cash_end":sides[-1][0], "living_trees_end":len(sides[-1][3]),
        "hire_price_end":prices[-1], "cash_meets_hire_price_end":sides[-1][0]>=prices[-1],
        "fundable_hire_with_base_and_capacity_end":
            (bool(sides[-1][1]) and sides[-1][0]>=prices[-1] and len(sides[-1][2])<conf["workerLimit"]) if end<game.turns else None,
    }
    assert set(target)==set(expected)
    for key,value in expected.items():
        if key in ("own_collector_delivery","tree_absorption"):
            if abs(target[key]-value) > 0.0000011:
                raise ValueError((key, target[key], value, t, end, physical))
        elif target[key]!=value:
            raise ValueError((key, target[key], value, t, end, physical))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-cache",type=Path)
    parser.add_argument("--out",type=Path,help="write a new dated coverage receipt, otherwise print JSON")
    parser.add_argument("--expect-games",type=int,help="assert an independently specified published waterline")
    args=parser.parse_args()
    indexes=registered_indexes()
    catalogs=[Corpus(index=i) for i in indexes]
    n_games=sum(len(c.rows) for c in catalogs)
    assert len(set.union(*(set(c.rows) for c in catalogs)))==n_games
    if args.expect_games is not None and n_games!=args.expect_games:
        raise ValueError(f"expected {args.expect_games} registered games, found {n_games}")
    counts={str(h):{"full":0,"censored":0} for h in (10,30)}
    by_batch={}
    cases=dict(CASES)
    for path,c in zip(indexes,catalogs):
        matched=set(c.rows)&set(cases)
        if not matched:  # Include an independent sample from every newly registered batch.
            nearest=min(c.rows.values(),key=lambda row:(int(row["transitions"]),row["id"]))
            stop=int(nearest["transitions"])
            cases[nearest["id"]]=tuple(sorted({t for t in (0,stop//2,stop-30,stop-10,stop-1) if 0<=t<stop}))
        subtotal={str(h):{"full":0,"censored":0} for h in (10,30)}
        for v in c.rows.values():
            n=int(v["transitions"])
            for h in (10,30):
                valid=max(0,n-h+1)*2
                for denominator in (counts,subtotal):
                    denominator[str(h)]["full"]+=valid
                    denominator[str(h)]["censored"]+=2*n-valid
        by_batch[path.relative_to(HERE).as_posix()]={"games":len(c.rows),"decision_seats":c.size,"coverage":subtotal}
    audit=[]
    for gid,roots in cases.items():
        game=open_game(gid,args.replay_cache)
        lab=FutureEconomy(game)
        rows=0; zero_cash=0; last_loss=0;came_back=0;equal_boundaries=0; notes=[]
        for t in roots:
            original=game.decision(t,0,lookbacks=())
            by_seat=[lab.labels(t,s) for s in (0,1)]
            assert original==game.decision(t,0,lookbacks=())
            for h in (10,30):
                stop=min(t+h,game.turns)
                ledger=visible_ledger(game._frames[t:stop+1],game._data["configuration"])[0]
                a,b=[v[str(h)] for v in by_seat]
                k="target" if t+h<=game.turns else "observed_to_terminal"
                assert a["observed_steps"]==stop-t and a["full_horizon"]==(t+h<=game.turns)
                assert a["terminal_at_observed_end"]==(None if stop!=game.turns else "scheduled" if stop==game._data["configuration"]["episodeSteps"]-1 else "early")
                assert a["target" if k!="target" else "observed_to_terminal"] is None
                assert a[k]["own"]==b[k]["opponent"] and a[k]["opponent"]==b[k]["own"]
                for s,side in enumerate(("own","opponent")):
                    v=a[k][side]
                    check_for_side(game,t,stop,s,v,ledger)
                    zero_cash+=int(v["own_collector_delivery"]==0)+int(v["tree_absorption"]==0)
                    last_loss+=v["first_loss_of_last_collector_offset"] is not None
                    came_back+=v["first_collector_present_again_offset"] is not None
                    rows+=1
                    if (gid.startswith("76a7") and t==104 and h==30 and not s==0) or (gid.startswith("6fe4") and t==84 and h==10 and s==0):
                        notes.append({"decision":t,"seat":s,"requested_horizon":h,"observed_steps":stop-t,"complete":a["full_horizon"],"observed":v})
                equal_boundaries+=int(t+h==game.turns)
        meta=next(c.rows[gid] for c in catalogs if gid in c.rows)
        audit.append(dict(id=gid,raw_sha256=meta["replay_sha256"],turns=game.turns,fixtures=roots,
                          verified_role_windows=rows,zero_income_fields=zero_cash,last_loss_windows=last_loss,
                          recovery_windows=came_back,exact_horizon_terminal_cases=equal_boundaries,examples=notes))
    report=dict(index_game_count=n_games,decision_seats=sum(c.size for c in catalogs),coverage=counts,coverage_by_batch=by_batch,
                coverage_method="Counts from the registered and hash-verified catalogs; direct cross-implementation values checked only for the listed fixtures.",
                fixtures=audit,absolute_cash_tolerance=0.0000011,
                source_hashes={f:sha256((HERE/f).read_bytes()).hexdigest() for f in (*(i.relative_to(HERE).as_posix() for i in indexes),"future_economy.py","future_economy_check.py")})
    encoded=json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)
    if args.out:
        args.out.write_text(encoded+"\n")
        print(json.dumps({"coverage":counts,"window_sides":sum(r["verified_role_windows"] for r in audit)}))
    else:
        print(encoded)


if __name__ == "__main__":
    main()
