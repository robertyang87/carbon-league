"""Reprice signed original games, pair physical seats and explain the first 500 cash lead of every win."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
BASE=STUDY.parents[1]
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
import official_game_replay as official
from turning_point_ledger import CashEvents,FastCarbon,sums_by_seat
from tree_win_gate import validate_examiner_configuration


def read(p):
    data=p.read_bytes()
    return json.loads(gzip.decompress(data) if p.suffix==".gz" else data)


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def require(b,msg):
    if not b:raise ValueError(msg)


def average(rows):
    return statistics.mean(rows)


def money(c):
    return {k:round(v,8) for k,v in sorted(c.items())}


def worker_state(obs):
    return {w:{"seat":s,"cell":v[0],"cargo":v[1],"role":v[2]} for s,p in enumerate(obs["players"]) for w,v in p[2].items()}


def terms_window(rows,cash,elapsed,seat,threshold):
    if threshold is None:
        # The biggest rise over each fixed horizon when the winner never had a lead >500.
        start=max(range(len(rows)-elapsed+1),key=lambda t:round(cash[t+elapsed]-cash[t],2))
        end=start+elapsed
    else:
        end=threshold;start=max(0,end-elapsed)
    sums=[Counter(),Counter()]
    for r in rows[start:end]:
        per=sums_by_seat(r["cash_events"])
        for s in (0,1):sums[s].update(per[s])
    values={k:sums[seat][k]-sums[1-seat][k] for k in set(sums[seat])|set(sums[1-seat])}
    require(abs(sum(values.values())-(cash[end]-cash[start]))<.00001,"window money mismatch")
    return {"horizon":elapsed,"start_state":start,"end_state":end,"start_lead":round(cash[start],2),
            "end_lead":round(cash[end],2),"lead_gain":round(cash[end]-cash[start],2),"focal_minus_enemy_by_term":money(values),
            "cash_by_physical_seat":[money(c) for c in sums],"step_rows":list(range(start,end))}


def replay(folder,row,off,arm,registered_hash):
    path=folder/row["raw_file"]; tape=read(path)
    require(tape["pack_sha256"]==registered_hash and digest(path)==row["raw_sha256"]==off["transcript_sha256"],"pack or transcript identity")
    validate_examiner_configuration(tape["configuration"])
    require(official.check(folder,row)==off and not tape["errors"] and not tape["audit_messages"],"failed independent official replay")
    court=tape["search_calls"]
    require([v[0] for v in court]==list(range(tape["steps"])) and any(v[1]=="searched" for v in court),"missing full original decisions")
    seed,seat=tape["seed"],tape["candidate_seat"]
    sim=FastCarbon(tape["game_configuration"]).reset(seed=seed)
    sim.cash=CashEvents(sim.cash)
    initial=list(sim.cash); sums=[Counter(),Counter()]
    leads=[initial[seat]-initial[1-seat]]
    trace=[]
    won=off["result"]=="win"
    for t,joint in enumerate(tape["actions"]):
        old=list(sim.cash)
        if won:workers=worker_state(sim.full_observation())
        sim.cash.events.clear();sim.step(*joint)
        by_seat=sums_by_seat(sim.cash.events)
        for s in (0,1):
            require(abs(sum(by_seat[s].values())-(sim.cash[s]-old[s]))<1e-7,"unexplained cash write")
            sums[s].update(by_seat[s])
        leads.append(sim.cash[seat]-sim.cash[1-seat])
        if won:
            after=worker_state(sim.full_observation())
            pieces={w:{"pre":v,"command":joint[v["seat"]].get(w,"NONE"),"post":after.get(w)}
                    for w,v in sorted(workers.items())}
            born={w:v for w,v in sorted(after.items()) if w not in workers}
            trace.append({"t":t,"cash_before":old,"cash_after":list(sim.cash),"joint":joint,"workers":pieces,
                          "born":born,"cash_events":list(sim.cash.events),"collisions":list(sim.last_collisions)})
    for s in (0,1):
        require(abs(sum(sums[s].values())+initial[s]-sim.cash[s])<1e-6,"whole game cash mismatch")
        sums[s]["terminal_rule_adjustment"]=off["official_reward_by_seat"][s]-sim.cash[s]
    item={"arm":arm,"seed":seed,"seat":seat,"terrain":off["map_sha256"],"transcript":str(path.relative_to(ROOT)),
          "raw_sha256":digest(path),"pack_sha256":registered_hash,"steps":off["steps"],"first_joint":tape["actions"][0],
          "original_search_returns":dict(Counter(v[1] for v in court)),"starting_cash":initial,
          "final_cash":list(sim.cash),"official_score_by_seat":off["official_reward_by_seat"],
          "result":off["result"],"margin":round(row["margin"],2),"by_seat":[money(c) for c in sums]}
    if won:
        cross=next((i for i,v in enumerate(leads) if v>500),None)
        windows=[terms_window(trace,leads,n,seat,cross) for n in (3,5,10) if len(trace)>=n]
        steps=sorted({t for w in windows for t in w["step_rows"]})
        item["turning_point"]={"first_strict_cash_lead_gt_500_state":cross,"peak_cash_lead":round(max(leads),2),
                               "fallback_if_never_above_500":"earliest maximal rise separately per horizon",
                               "windows":windows,"turns":[trace[t] for t in steps]}
    return item


def paired(games,maps):
    contrasts=[]
    for seed in maps:
        for s in (0,1):
            p,c=([g for g in games if (g["seed"],g["seat"],g["arm"])==(seed,s,a)][0] for a in ("parent","candidate"))
            require(p["terrain"]==c["terrain"],"map alias")
            terms=[]
            for i in (s,1-s):
                terms.append({k:c["by_seat"][i].get(k,0)-p["by_seat"][i].get(k,0) for k in set(c["by_seat"][i])|set(p["by_seat"][i])})
            margins=[x["official_score_by_seat"][s]-x["official_score_by_seat"][1-s] for x in (p,c)]
            delta=margins[1]-margins[0]
            require(abs(delta-sum(terms[0].values())+sum(terms[1].values()))<.0001,"focal and enemy differences fail to close")
            contrasts.append({"seed":seed,"seat":s,"parent_margin":round(margins[0],2),"candidate_margin":round(margins[1],2),
                              "delta":round(delta,2),"own_delta":round(sum(terms[0].values()),2),"enemy_delta":round(sum(terms[1].values()),2),
                              "first_command_equal_by_seat":[c["first_joint"][i]==p["first_joint"][i] for i in (0,1)],
                              "own_term_delta":money(terms[0]),"enemy_term_delta":money(terms[1])})
    clusters=[average(x["delta"] for x in contrasts if x["seed"]==s) for s in maps]
    m=average(clusters);se=statistics.stdev(clusters)/(len(maps)**.5)
    require(len(maps)==2,"t reference fixed to two map clusters")
    result={"pairs":contrasts,"map_deltas":dict(zip(maps,clusters,strict=True)),"mean_margin_delta":m,
            "by_physical_seat":[average(x["delta"] for x in contrasts if x["seat"]==s) for s in (0,1)],
            "se_by_two_maps":se,"two_map_t95_reference":[m-12.7062047364*se,m+12.7062047364*se],
            "t_interval_note":"Two known or deliberately reserved development maps, one residual df; illustrative sensitivity only. Different original search RNG streams and independent timed searches.",
            "arms":{a:{"wins":sum(g["result"]=="win" for g in games if g["arm"]==a),
                        "draws":sum(g["result"]=="draw" for g in games if g["arm"]==a),
                        "mean_margin":average(g["margin"] for g in games if g["arm"]==a)} for a in ("parent","candidate")}}
    for field in ("own_term_delta","enemy_term_delta"):
        keys=set().union(*(v[field].keys() for v in contrasts))
        result["mean_"+field]=money({k:average(v[field].get(k,0) for v in contrasts) for k in keys})
    return result


def run(kind):
    historical=kind=="historical";src=STUDY if historical else STUDY/"replication"
    plan=read(src/"protocol.json");rel=read(STUDY/"release.json")
    comparison=read(src/"comparison.json")
    maps=plan["maps"] if historical else plan["new_dev"]["seeds"]
    require(len(maps)==2 and len(set(maps))==2,"unregistered map list")
    games=[]
    for seed in maps:
        for s in (0,1):
            for arm in ("parent","candidate"):
                if historical and arm=="parent":
                    folder=STUDY.parent/plan["parent_source"][str(seed)]
                    rows=read(folder/"results.json")["games"]
                    row,=[v for v in rows if (v["seed"],v["candidate_seat"])==(seed,s)]
                    receipt,=[v for v in read(folder/"official_replay.json")["runs"][folder.name]["games"] if (v["seed"],v["seat"])==(seed,s)]
                else:
                    folder=(src/"native" if historical else src/"games")/arm
                    signed=read(folder/f"receipt_{seed}_{s}.json")
                    row,receipt=signed["game"],signed["official"]
                ref=rel["candidate_sha256"] if arm=="candidate" else rel["input_sha256"]["data/weights/sapling_1200k/terminal_return_fit_v1.py"]
                games.append(replay(folder,row,receipt,arm,ref))
    assert len(games)==len({(r["arm"],r["seed"],r["seat"]) for r in games})==8
    stats=paired(games,maps)
    if historical:
        require(abs(comparison["mean_paired_change"]-stats["mean_margin_delta"])<1e-6,"historical registered comparison")
    else:
        income=read(src/"income.json")
        records=income["games"] if isinstance(income,dict) else income
        for g in games:
            saved,=[x for x in records if (x["arm"],x["seed"],x["seat"])==(g["arm"],g["seed"],g["seat"])]
            require(saved["raw_sha256"]==g["raw_sha256"] and saved["official_score_by_seat"]==g["official_score_by_seat"],"independent confirmation failed")
            for i in (0,1):
                require(all(abs(g["by_seat"][i].get(k,0)-v)<1e-6 for k,v in saved["by_seat"][i].items()),"independent cash audit differs")
    out={"cohort":kind,"scope":"historical controls played earlier on the same maps and seats" if historical else "separate newer development pair, contemporary parent and candidate, independent opponent search",
         "protocol_sha256":digest(src/"protocol.json"),"comparison_sha256":digest(src/"comparison.json"),"script_sha256":digest(Path(__file__)),
         "cash_event_scanner_sha256":digest(BASE/"turning_point_ledger.py"),"official_verifier_sha256":digest(BASE/"official_game_replay.py"),
         "games":games,"statistics":stats}
    path=HERE/(kind+".json.gz")
    packed=gzip.compress(json.dumps(out,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode(),mtime=0)
    if path.exists() and path.read_bytes()!=packed:raise ValueError("different previous economic receipt")
    temp=path.with_suffix(".tmp");temp.write_bytes(packed);temp.replace(path)
    print(kind,len(games),stats["mean_margin_delta"],digest(path),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort",choices=("historical","replication"))
    run(parser.parse_args().cohort)
