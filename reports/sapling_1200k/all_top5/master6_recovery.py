"""Count solvent and insolvent collector gaps on a frozen common-opponent/physical-seat panel."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from context import Corpus, HERE, ROOT, RULE_FILE
from economy_extract import KEYS, reconstruct
from defense import DEST, MOVE

ARMS = ("master6", "other_top5")
FLAGS = ("strict", "gap_after_collector", "gap_tree", "gap_affordable_hire", "gap_only_funded_planter")
CUTS = (1, 5, 10, 30, 100)
PANEL = HERE / "appendix_4bfdabf5/master6_opponent_controls.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample(game, t, seat):
    cfg=game._data["configuration"]
    raw=game._obs(t,seat,terminal=True)
    cash,bases,workers,trees=raw["players"][seat]
    collectors=sum(v[2]=="COLLECTOR" for v in workers.values())
    planters=len(workers)-collectors
    if any(x[1]!=0 for x in workers.values() if x[2]=="PLANTER"):
        raise ValueError("unexpected planter inventory")
    # These prices are constant on the archived panel; an unrecognised regime must be reviewed.
    if cfg["recCollectorIncreaseCost"] or cfg["recPlanterIncreaseCost"] or cfg["playerTreeProtectiveNumber"]<0:
        raise ValueError("unhandled variable recruitment/first-tree price")
    if not bases:
        raise ValueError("panel unexpectedly lacks a base")
    can_hire=cash>=cfg["recCollectorCost"] and len(workers)<cfg["workerLimit"]
    can_fund_tree=not trees and planters>0 and cash>=min(cfg["plantCost"],cfg["seizeCost"])
    lower_price=min(cfg["plantCost"],cfg["seizeCost"],cfg["recCollectorCost"],cfg["recPlanterCost"])
    strict=bool(planters and not collectors and not trees and cash<lower_price)
    return dict(cash=cash,collectors=collectors,planters=planters,trees=len(trees),
                tree_income=sum(raw["trees"][uid][1] for uid in trees),can_hire_collector=can_hire,
                can_fund_first_tree_or_seizure=bool(can_fund_tree),strict=strict)


def ranges(flags):
    start=None
    for i,b in enumerate([*flags,False]):
        if b and start is None:start=i
        elif not b and start is not None:
            yield start,i
            start=None


def intervals(states,flags):
    turns=len(states)-1
    return [dict(start=a,end_exclusive=b,decisions=b-a,starts_with=states[a],
                 after=states[b],reaches_last_settlement=(b==turns))
            for a,b in ranges(flags)]


def analyze(corpus,gid,seat,opponent,arm):
    game=corpus.open(gid)
    states=[sample(game,t,seat) for t in range(game.turns+1)]
    seen=False
    flags={k:[] for k in FLAGS}
    for p in states[:-1]:
        seen=seen or p["collectors"]>0
        gap=seen and p["collectors"]==0
        for k,v in {"strict":p["strict"],"gap_after_collector":gap,"gap_tree":gap and p["trees"]>0,
                    "gap_affordable_hire":gap and p["can_hire_collector"],
                    "gap_only_funded_planter":gap and p["trees"]==0 and p["can_fund_first_tree_or_seizure"]
                                          and not p["can_hire_collector"]}.items():
            flags[k].append(bool(v))
    # Count again from the official raw shared roster and stored per-seat ACTIVE flag.
    # Track the set of collector IDs ever seen independently of sample()'s running count.
    ever_collector_ids=set()
    strict_root_ids=set()
    gap_root_ids=set()
    for t,pair in enumerate(game._frames[:-1]):
        if pair[seat]["status"]!="ACTIVE":
            raise ValueError("unexpected inactive decision")
        money,_,workers,trees=pair[0]["observation"]["players"][seat]
        current={uid for uid,data in workers.items() if data[2]=="COLLECTOR"}
        ever_collector_ids.update(current)
        if ever_collector_ids and not current:
            gap_root_ids.add(t)
        if workers and all(data[2]=="PLANTER" and data[1]==0 for data in workers.values()) and not trees and money<20:
            strict_root_ids.add(t)
    if strict_root_ids!={t for t,v in enumerate(flags["strict"]) if v} or gap_root_ids!={t for t,v in enumerate(flags["gap_after_collector"]) if v}:
        raise ValueError("independent raw-roster check disagrees with the rule-based scan")
    runs={k:intervals(states,v) for k,v in flags.items()}
    ledger=reconstruct(game._frames,game._data["configuration"])[0] if strict_root_ids else None
    for episode in runs["gap_after_collector"]:
        a,b=episode["start"],episode["end_exclusive"]
        paid=sum(x["tree_income"] for x in states[a+1:b+1])
        episode["actual_tree_income_until_exit_or_end"]=round(paid,5)
        episode["collector_returned_at_end"]=states[b]["collectors"]>0
        episode["cash_min_max"]=[min(p["cash"] for p in states[a:b+1]),max(p["cash"] for p in states[a:b+1])]
        episode["first_new_tree"] = next((k for k in range(a+1,b+1) if states[k-1]["trees"]==0 and states[k]["trees"]>0),None)
    for episode in runs["strict"]:
        a,b=episode["start"],episode["end_exclusive"]
        if a:
            before=game._frames[a-1][0]["observation"]["players"]
            after=game._frames[a][0]["observation"]["players"]
            commands=[entry["action"] or {} for entry in game._frames[a]]
            lc=[]
            for uid,(pos,load,kind) in before[seat][2].items():
                if kind!="COLLECTOR":continue
                end=DEST[pos][MOVE[commands[seat].get(uid)]]
                colocated=[dict(seat=side,id=w,role=role,cargo=c,source=p)
                           for side in (0,1) for w,(p,c,role) in before[side][2].items()
                           if w!=uid and DEST[p][MOVE[commands[side].get(w)]]==end]
                lc.append(dict(id=uid,cargo=load,pos=pos,destination=end,existing_workers_arriving=colocated,
                               enemy_base=end in before[1-seat][1].values()))
            lt=[dict(id=uid,position=pos,age=age,transferred_to_opponent=uid in after[1-seat][3],
                     at_lifespan=age>=game._data["configuration"]["treeLifespan"])
                for uid,(pos,age) in before[seat][3].items()]
            episode["immediately_before_entry"]={"state":states[a-1],"collectors_lost":lc,"trees_lost":lt}
        episode["gross_cash_event_absolutes"]=dict(zip(KEYS,
             [round(sum(abs(ledger[t][seat][i]) for t in range(a,b)),7) for i in range(len(KEYS))]))
        episode["strict_at_terminal"]=b==game.turns and states[b]["strict"]
        episode["ever_had_collector_before_entry"]=any(p["collectors"] for p in states[:a])
        episode["positive_cash_increments"]=[dict(t=k,from_cash=states[k-1]["cash"],to_cash=states[k]["cash"]) for k in range(a+1,b+1) if states[k]["cash"]>states[k-1]["cash"]]
        episode["status_at_end"]="terminal" if b==game.turns else "active"
    views=[dict(t=k,**states[k]) for k in (0,game.turns)]
    cfg=game._data["configuration"]
    return dict(id=gid,opponent=opponent,arm=arm,seat=seat,replay_sha256=corpus.rows[gid]["replay_sha256"],turns=game.turns,
                prices={k:cfg[k] for k in ("recCollectorCost","recPlanterCost","plantCost","seizeCost","workerLimit")},
                end_states=views,runs=runs)


def summary(rows):
    indexed=defaultdict(lambda:defaultdict(list))
    for r in rows:indexed[r["opponent"],r["seat"]][r["arm"]].append(r)
    stats={}
    for arm in ARMS:
        sample_rows=[x for x in rows if x["arm"]==arm]
        rstats={}
        for name in FLAGS:
            all_runs=[(r,x) for r in sample_rows for x in r["runs"][name]]
            rstats[name]={"episodes":len(all_runs),"seat_decisions":sum(e["decisions"] for _,e in all_runs)}
            for n in CUTS:
                keep=[(r,e) for r,e in all_runs if e["decisions"]>=n]
                games={r["id"] for r,_ in keep}
                rg=defaultdict(list)
                for (op,_),arms in indexed.items():
                    rg[op].append(sum(r["id"] in games for r in arms[arm])/len(arms[arm]))
                rstats[name][str(n)]=dict(games=len(games),opponents=len({r["opponent"] for r,e in keep}),
                     episodes=len(keep),equal_opponent_rate=sum(sum(x)/len(x) for x in rg.values())/len(rg))
        stats[arm]=dict(games=len(sample_rows),opponents=len({r["opponent"] for r in sample_rows}),
                        active_decisions=sum(r["turns"] for r in sample_rows),flags=rstats)
    return stats


def main():
    panel=json.loads(PANEL.read_bytes())["groups"]["21_50"]
    corpora=[Corpus(),Corpus(manifest=HERE/"appendix_499886f1/manifest.json"),
                     Corpus(manifest=HERE/"appendix_4bfdabf5/manifest.json")]
    catalog={gid:c for c in corpora for gid in c.order}
    rows=[]
    for layer in panel["strata"]:
        for arm,ids in zip(ARMS,layer["arm_ids"],strict=True):
            for gid in ids:
                c=catalog[gid];s=layer["seat"];name=layer["opponent"]
                meta=c.provenance(gid)
                if meta["seat_names"][s]!=name or (meta["seat_names"][1-s]=="claude-AlphaCarbon-Master6")!=(arm==ARMS[0]):
                    raise ValueError("wrong opponent or physical seat")
                rows.append(analyze(c,gid,s,name,arm))
    if len({r["id"] for r in rows})!=309 or Counter(r["arm"] for r in rows)!=dict(zip(ARMS,(81,228))):
        raise ValueError("covered map counts do not match frozen panel")
    data={"panel":"Same registered external heavy rank 21–50 player and physical seat, 44 strata, 26 players",
          "definition":"Strict = an active turn with at least one planter, zero collectors, zero owned trees, zero planter cargo, cash below both recruitment prices, first tree and seizure prices. Does not exclude involuntary receipts at one's own base. Elapsed consecutive steps are half-open [start,end_exclusive); final settlement is shown separately.",
          "source_sha256":{p.name:digest(p) for p in (Path(__file__),PANEL,HERE/"context.py",HERE/"defense.py",HERE/"economy_extract.py",HERE.parent/"turning_point_ledger.py",ROOT/"code/fastsim.py",RULE_FILE.with_name("helpers.py"))},
          "summary":summary(rows),"maps":rows}
    (HERE/"master6_recovery.json").write_text(json.dumps(data,ensure_ascii=False,indent=1,allow_nan=False)+"\n")
    print(json.dumps(data["summary"],ensure_ascii=False))


if __name__=="__main__":
    main()
