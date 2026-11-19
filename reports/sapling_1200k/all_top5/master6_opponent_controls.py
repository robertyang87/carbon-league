"""Compare the same external ladder opponent in the same physical seat against Master6 and other leaders."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
APPENDIX = HERE / "appendix_499886f1"
MASTER = "claude-AlphaCarbon-Master6"
M5 = "claude-AlphaCarbon-Master5s"
METRICS = ("bank", "tree", "bank_plus_tree", "cash", "workers_lost", "hires", "workers",
           "owned_trees", "seized_enemy_trees", "lost_trees")
PHASES = ((0,100),(100,200))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load():
    seals = {}
    def read(path):
        raw = path.read_bytes()
        seals[str(path.relative_to(HERE.parent))] = digest(raw)
        return gzip.decompress(raw) if path.name.endswith(".gz") else raw
    board = json.loads(read(HERE.parent / "latest5_inventory_sources.json.gz"))
    leaderboard = board["leaderboard"]
    by_id = {v["id"]:v for v in leaderboard["strategies"]+leaderboard["extra"]}
    name_counts = Counter(v["name"] for v in by_id.values())
    ranks = {v["name"]:v["board_rank"] for v in by_id.values()
             if name_counts[v["name"]]==1 and v.get("board")=="heavy" and isinstance(v.get("board_rank"),int)}
    cards = {}
    for where in (HERE, APPENDIX):
        man = json.loads(read(where / "manifest.json"))
        for g in man["games"]:
            if g["id"] in cards:
                raise ValueError("repeated game ID")
            cards[g["id"]]=g
    if len({v["seed"] for v in cards.values()}) != len(cards):
        raise ValueError("duplicate map")
    layout=json.loads(read(HERE / "layout.json.gz"))
    fields={n:i for i,n in enumerate(layout["field_order"])}
    layouts={r["id"]:r for r in layout["games"]}
    econ={}
    for path in sorted((HERE/"economy_data").glob("*.jsonl.gz")):
        for line in read(path).splitlines():
            r=json.loads(line); gid=r["id"]
            if gid in econ or cards[gid]["sha256"]!=r["replay_sha256"]:
                raise ValueError("cash identity")
            econ[gid]=r["record"]
    defense={}
    for line in read(HERE/"defense_traces.jsonl.gz").splitlines():
        r=json.loads(line);gid=r["id"]
        if gid in defense or cards[gid]["sha256"]!=r["sha256"]:raise ValueError("defense identity")
        defense[gid]=r
    for r in json.loads(read(APPENDIX/"cases.json.gz"))["rows"]:
        gid=r["id"]
        if gid in econ or r["source_sha256"]!=cards[gid]["sha256"]:raise ValueError("appendix identity")
        econ[gid],layouts[gid],defense[gid]=r["economy"],r["layout"],r["defense"]
    if set(cards)!=set(econ) or set(cards)!=set(layouts) or set(cards)!=set(defense):
        raise ValueError("incomplete evidence")
    exclusion=Counter(); result=[]
    for gid,g in sorted(cards.items()):
        if g["cohort"]!="external" or not g["clean_terminal"]:
            exclusion["mutual_or_unclean"]+=1;continue
        seat=g["seat_strategy_ids"].index(None)
        name=g["seat_names"][seat];head=g["seat_names"][1-seat]
        rank=ranks.get(name)
        if rank is None:
            exclusion["rank_or_identity_unconfirmed"]+=1;continue
        if rank<=5:raise ValueError("rank and cohort disagree")
        e,layout,defn=econ[gid],layouts[gid],defense[gid]
        if layout["sha256"]!=g["sha256"] or e["rewards"]!=g["score_by_seat"]:
            raise ValueError(gid)
        turns=g["transitions"];totals=defn["seat_totals"][seat]
        # Every whole-game hire reconstructed independently from actual commands
        # and both rosters must reconcile exactly to the cash ledger at this fee.
        hires_total=-e["whole"]["seats"][seat]["recruit"]/30
        if abs(hires_total-totals.get("recruits",0))>1e-5: raise ValueError("hire fee " + gid)
        end_workers=totals.get("recruits",0)-totals.get("collision_deaths",0)
        if not 0<=end_workers<=10:raise ValueError("worker reconciliation "+gid)
        def stock(t):
            if t==0:return [None,0,0]
            if str(t) in layout["snapshots"]:
                row=layout["snapshots"][str(t)][seat]
                return [row[fields["cash"]],row[fields["collectors"]]+row[fields["planters"]],row[fields["trees"]]]
            if t<turns:raise ValueError("missing stock snapshot "+gid)
            return [e["cash_end"][seat],end_workers,e["terminal_assets"]["seats"][seat]["final_trees"]]
        values=[]
        for a,b in PHASES:
            segment=next((s for s in e["stages"] if s["a"]==a),None)
            f=segment["seats"][seat] if segment else {}
            if segment and segment["b"]!=min(turns,b) or not segment and turns>a:
                raise ValueError("phase endpoint " + gid)
            start=stock(a);end=stock(b);hires=-f.get("recruit",0)/30
            deaths=hires+start[1]-end[1]
            if abs(hires-round(hires))>1e-5 or deaths<0:raise ValueError("phase workers "+gid)
            seiz=[x[1] for x in defn["tree_seizures"] if a<x[0]<=b]
            v=dict(bank=f.get("bank",0),tree=f.get("tree",0),cash=end[0],hires=round(hires),
                   workers_lost=round(deaths),workers=end[1],owned_trees=end[2],
                   seized_enemy_trees=seiz.count(seat),lost_trees=seiz.count(1-seat))
            v["bank_plus_tree"]=v["bank"]+v["tree"]
            values.append([v[k] for k in METRICS])
        result.append(dict(id=gid,seed=g["seed"],seat=seat,opponent=name,rank=rank,
                           head=head,turns=turns,final=g["score_by_seat"],phase_values=values))
    return result,exclusion,seals,board["as_of_board_utc"],max(ranks.values())


def prepare(rows, low,high, physical=None, control=None, complete=False):
    include=[g for g in rows if low<=g["rank"]<=high and (physical is None or g["seat"]==physical)
             and (control is None or g["head"] in (MASTER,control)) and (not complete or g["turns"]>=200)]
    groups=defaultdict(lambda:defaultdict(list))
    for g in include:groups[g["opponent"],g["seat"]][g["head"]==MASTER].append(g)
    matched={k:v for k,v in groups.items() if v[True] and v[False]}
    byname=defaultdict(list)
    for (name,seat),arms in sorted(matched.items()):
        byname[name].append((seat,arms))
    tally=Counter(g["head"]==MASTER for g in include)
    return include,groups,matched,byname,[tally[True],tally[False]]


def summarize(rows, low, high, physical=None, control=None, complete=False, bootstrap=False, examples=False):
    pool,groups,matched,names,available=prepare(rows,low,high,physical,control,complete)
    count=[sum(len(v[b]) for v in matched.values()) for b in (True,False)]
    result=dict(available_maps=available,available_opponents=len({g["opponent"] for g in pool}),matched_maps=count,
                matched_opponents=len(names),matched_opponent_seats=len(matched),both_seats=sum(len(v)==2 for v in names.values()),
                single_m6_map_strata=sum(len(v[True])==1 for v in matched.values()),
                comparator_heads=dict(Counter(g["head"] for v in matched.values() for g in v[False])),
                early_end_ids=[g["id"] for v in matched.values() for arm in (True,False) for g in v[arm] if g["turns"]<200])
    if not names:return result
    comps=[]; strata=[]; names_data=[]
    for n,seats in sorted(names.items()):
        avgs=[]
        for s,arms in seats:
            means=np.asarray([np.mean([g["phase_values"] for g in arms[b]],axis=0) for b in (True,False)])
            avgs.append(means)
            strata.append(dict(opponent=n,seat=s,arm_ids=[[r["id"] for r in arms[b]] for b in (True,False)],
                               means=means.round(5).tolist()))
            if len(arms[True])>=2 and len(arms[False])>=2:
                k=METRICS.index("bank_plus_tree"); eff=means[0,1,k]-means[1,1,k]
                def rep(b, target):
                    return min(arms[b],key=lambda g:(abs(g["phase_values"][1][k]-target),g["id"]))
                comps.append((eff,n,s,rep(True,means[0,1,k]),rep(False,means[1,1,k])))
        names_data.append(np.mean(avgs,axis=0))
    arr=np.asarray(names_data); mu=arr.mean(axis=0)
    diff=arr[:,0]-arr[:,1]
    result.update(master6_means=mu[0].round(4).tolist(),others_means=mu[1].round(4).tolist(),
                  differences=diff.mean(axis=0).round(4).tolist(),
                  names_with_lower_values=(diff<0).sum(axis=0).tolist(),
                  names_with_higher_values=(diff>0).sum(axis=0).tolist())
    if bootstrap:
        rng=np.random.default_rng(20261119)
        sampled=diff[rng.integers(0,len(arr),size=(10000,len(arr)))].mean(axis=1)
        result["opponent_cluster_interval"]=np.quantile(sampled,[.025,.975],axis=0).round(4).tolist()
    if examples:
        result["strata"]=strata
        for key, sign in (("counterexample",1),("suppressed_example",-1)):
            suitable=[x for x in comps if x[0]*sign>0]
            if suitable:
                eff,n,s,ma,other=max(suitable,key=lambda x:(x[0]*sign,x[1],x[2]))
                result[key]=dict(midgame_cashflows_group_difference=round(float(eff),4),opponent=n,seat=s,
                                 master6_game=ma,control_game=other)
    return result


def main():
    rows,excluded,seals,when,high=load()
    groups={}
    for name,lo,hi in (("21_50",21,50),("6_20",6,20),("51_plus",51,high)):
        groups[name]=summarize(rows,lo,hi,bootstrap=True,examples=name=="21_50")
    group=groups["21_50"]
    group["by_seat"]={str(s):summarize(rows,21,50,physical=s) for s in (0,1)}
    group["fixed_comparator"]={h:summarize(rows,21,50,control=h) for h in sorted({g["head"] for g in rows}-{MASTER})}
    group["survived_to_200"]=summarize(rows,21,50,complete=True)
    output=dict(rank_time=when,evidence_sha256=seals,metrics=METRICS,phases=PHASES,
                rates_definition="Equal weight per matched external opponent, same physical seat only. Within that opponent weight each eligible seat equally. Within a seat, average all available maps of each arm. All differences are from the external player's perspective, Master6 minus other leaders.",
                finite_horizon="Flow after official termination equals zero. For t100/t200 beyond termination, hold final cash, workers and owned trees. Also report the common-support subset lasting >=200.",
                interval="10,000 opponent-cluster resamples; retained games and seats per sampled identity remain fixed.",
                excluded=dict(excluded),groups=groups)
    (HERE/"master6_opponent_controls.json").write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
