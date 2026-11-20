"""Map-weighted comparisons by physical seat, named player, model arm, batch and outcome."""
from collections import Counter, defaultdict
import gzip
import json
import hashlib
import zipfile
from pathlib import Path
from statistics import mean,median

HERE=Path(__file__).resolve().parent
TICKS=(30,60,100,150,200,250,299)
THRESHOLDS=dict(cash=100,delivery=100,ground_rent=100,tree_investment=100,tree_net=100,
                workers=1,collectors=1,planters=1,trees=1,cargo=100,cargo_distance=1,
                one_or_more_threatened=.10,all_routes_threatened=.10,choice_of_hot_and_clear=.10,
                screen_plus_separate_clear=.10,takes_clear_when_choice=.10,lost_on_threatened_home_move=.10)
DENOM=dict(one_or_more_threatened="loaded_turns",all_routes_threatened="loaded_turns",
           multiple_home_lanes="loaded_turns",choice_of_hot_and_clear="loaded_turns",
           screen_plus_separate_clear="loaded_turns",takes_clear_when_choice="choice_of_hot_and_clear",
           screens_and_takes_clear="screen_plus_separate_clear",lost_on_threatened_home_move="one_or_more_threatened")
def outcome(game):
    m=game["meta"]; scores=m["official_scores_by_seat"]; seat=m["seat"]
    return "W" if scores[seat]>scores[1-seat] else "L" if scores[seat]<scores[1-seat] else "D"

def groups(games,field):
    out=defaultdict(list)
    for g in games:
        out[field(g)].append(g)
    return out

def by_role(g,role):
    return g["meta"]["seat"] ^ role

def report_part(games):
    maps=groups(games,lambda g:g["meta"]["terrain"])
    result={"n_games":len(games),"n_maps":len(maps),"outcomes":dict(Counter(map(outcome,games))),"ticks":{}}
    for t in map(str,TICKS):
        beds={k:[g for g in gg if t in g["ticks"]] for k,gg in maps.items()}
        keys=sorted(set(next(iter(g["ticks"][t]))["state"].keys()) | set(DENOM)) if (g:=next((g for g in games if t in g["ticks"]),None)) else []
        metrics={}
        for field in keys:
            vals=[]
            for terrain,gg in beds.items():
                if not gg: continue
                arms=[]
                for side in (0,1):
                    snapshots=[g["ticks"][t][by_role(g,side)] for g in gg]
                    if field in DENOM:
                        c=Counter()
                        for s in snapshots:
                            c.update(s["previous_window_opportunities"])
                        val=c[field]/c[DENOM[field]] if c[DENOM[field]] else None
                    elif field=="cargo_distance":
                        loads=sum(s["state"]["cargo"] for s in snapshots)
                        val=sum(s["state"]["cargo_steps"] for s in snapshots)/loads if loads else None
                    else:
                        val=mean(s["state"][field] for s in snapshots)
                    arms.append(val)
                if None not in arms:
                    vals.append([*arms,arms[0]-arms[1]])
            metrics[field]=dict(physical_maps=len(vals),own=mean(v[0] for v in vals) if vals else None,
                examiner=mean(v[1] for v in vals) if vals else None,within_game_gap=mean(v[2] for v in vals) if vals else None,
                median_map_gap=median(v[2] for v in vals) if vals else None)
        result["ticks"][t]=metrics
    return result

def dump(path,obj):
    path.write_bytes(gzip.compress(json.dumps(obj,separators=(",",":"),allow_nan=False).encode(),mtime=0))

def clocks(games):
    # Read only the three verified archives already used for the profiles.
    # "elapsed" is the external request. The two public_* fields are agent process clocks.
    per_batch=defaultdict(lambda:defaultdict(list))
    for g in games:
        m=g["meta"]; b=m["batch"]
        path=HERE.parent/b/"hosted"/(str(m["archive_id"])+".zip")
        with zipfile.ZipFile(path) as z:
            raw=z.read(m["actor"]+"/turns.jsonl")
        assert hashlib.sha256(raw).hexdigest()==m["turns_sha256"]
        prev=0
        for line in raw.splitlines():
            r=json.loads(line); c=r["calls"][1-m["seat"]]; a=c["response"]; n=a["search_counts"]["aborted"]
            assert n-prev in (0,1)
            end=next(t for t in TICKS if r["step"]<t)
            v=(a["search_last"]["candidates"],a["search_last"]["depth"],n-prev,
               c["elapsed"],a["public_wall_seconds"],a["public_cpu_seconds"])
            per_batch[b][end].append(v)
            per_batch["all"][end].append(v)
            prev=n
    out={}
    for group,periods in per_batch.items():
        out[group]={}
        for end,values in periods.items():
            k,h,cut,ext,wall,cpu=zip(*values)
            cells=sorted(ext)
            out[group][end]=dict(calls=len(k),single=k.count(1),positive_depth=sum(v>0 for v in h),
                multizero=sum(a>1 and b==0 for a,b in zip(k,h)),cutoffs=sum(cut),
                candidates=dict(Counter(k)),depths=dict(Counter(h)),
                external_p95=cells[(95*len(ext)+99)//100-1],external_max=max(ext),external_slow=sum(t>1 for t in ext),
                external_seconds=sum(ext),agent_wall_seconds=sum(wall),agent_cpu_seconds=sum(cpu))
    return out

def main():
    get=lambda side:json.loads(gzip.decompress((HERE/(side+"_profile.json.gz")).read_bytes()))
    top,sap=get("top5"),get("sapling")
    assert (len(top),len(sap))==(30,44) and len({g["meta"]["id"] for g in top+sap})==74
    sets=[set(x["meta"]["terrain"] for x in gg) for gg in (top,sap)]
    assert len(sets[0])==30 and len(sets[1])==10 and not (sets[0]&sets[1])
    from sapling import ledger
    config=ledger.config()
    for g in top:
        a=g["meta"]["configuration"]
        assert {k:v for k,v in config.items() if k not in {"agentTimeout","actTimeout","runTimeout","randomSeed"}}=={
            k:v for k,v in a.items() if k not in {"agentTimeout","actTimeout","runTimeout","randomSeed"}}
    fields={"seat":lambda g:g["meta"]["seat"],"actor":lambda g:g["meta"]["actor"],"result":outcome,
        "batch_arm":lambda g:g["meta"].get("batch","ladder")+"/"+g["meta"]["actor"],
        "seat_result":lambda g:f"{g['meta']['seat']}/{outcome(g)}","terrain":lambda g:g["meta"]["terrain"]}
    data={}
    for name,gg in (("top",top),("sap",sap)):
        data[name]={"all":report_part(gg),
            **{k:{label:report_part(s) for label,s in groups(gg,key).items()} for k,key in fields.items()}}
    cad={}
    for k,tol in THRESHOLDS.items():
        diffs=[]
        for t in map(str,TICKS[:6]):
            tv=data["top"]["all"]["ticks"][t][k]["within_game_gap"]
            sv=data["sap"]["all"]["ticks"][t][k]["within_game_gap"]
            diffs.append(None if tv is None or sv is None else tv-sv)
        first=next((TICKS[i] for i in range(5) if diffs[i] is not None and diffs[i+1] is not None and
                    abs(diffs[i])>=tol and abs(diffs[i+1])>=tol and diffs[i]*diffs[i+1]>0),None)
        cad[k]=dict(descriptive_cut=tol,first_of_two_same_direction=first,difference=diffs)
    data["sapling_examiner_clock_by_period"]=clocks(sap)
    data["ladder_examiner_measured_clock"]=None
    panel_path=HERE.parents[1]/"broader_elite_backbone_dev/panel.json"
    raw=panel_path.read_bytes()
    panel=json.loads(raw)["games"]
    overlaps={}
    for name,terrains in zip(("top","sap"),sets,strict=True):
        overlaps[name]=[{k:g[k] for k in ("id","terrain_sha256","split","seat","teacher")}
                        for g in panel if g["terrain_sha256"] in terrains]
    data["broader_panel_overlap"]=dict(panel_sha256=hashlib.sha256(raw).hexdigest(),
        partitions=dict(Counter(g["split"] for g in panel)),overlap=overlaps)
    balanced={}
    for t in map(str,TICKS):
        balanced[t]={}
        for k in THRESHOLDS:
            rates=[[data[c]["seat"][seat]["ticks"][t][k]["within_game_gap"] for seat in (0,1)] for c in ("top","sap")]
            balanced[t][k]=mean(rates[0])-mean(rates[1]) if all(None not in r for r in rates) else None
    data["seat_equal_between_cohorts"]=balanced
    data["between_cohorts"]=cad
    data["rule_equality_excluding_seed_and_time_controls"]=True
    dump(HERE/"map_comparison.json.gz",data)
    for k,v in cad.items():
        print(k,v["first_of_two_same_direction"],[None if n is None else round(n,2) for n in v["difference"]])
if __name__=="__main__":
    main()
