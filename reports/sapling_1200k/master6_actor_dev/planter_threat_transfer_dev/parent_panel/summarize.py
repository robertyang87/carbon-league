"""Count opportunities, exposures and actual cash/outcomes with the map as the unit of uncertainty."""
from collections import Counter,defaultdict
from statistics import mean
import random

from panel import HERE,check,read,save,sha


def examined(rows):
    result={"roots":len(rows),"terrains":len({r["map"] for r in rows}),"games":len({(r["map"],r["seat"]) for r in rows})}
    for side in ("parent","pay"):
        result[side+"_survived"]=sum(r[side]["survived"] for r in rows)
        result[side+"_charged"]=sum(r[side]["worker_debit"]>0 for r in rows)
        result[side+"_expense"]=sum(r[side]["worker_debit"] for r in rows)
        result[side+"_own_tree_after"]=sum((r[side]["tree_at_proposed_cell_after"] or {}).get("side")==r["seat"] for r in rows)
    result.update(parent_survived_pay_died=sum(r["parent"]["survived"] and not r["pay"]["survived"] for r in rows),
                  parent_died_pay_survived=sum(not r["parent"]["survived"] and r["pay"]["survived"] for r in rows),
                  parent_and_pay_died=sum(not r["parent"]["survived"] and not r["pay"]["survived"] for r in rows),
                  reachable_by_existing=sum(bool(r["public_reach"]["existing"]) for r in rows),
                  enemy_actually_arrived=sum(bool(r["pay"]["enemy_real_existing_arrivals"]) for r in rows),
                  enemy_spawn_possible=sum(bool(r["public_reach"]["possible_enemy_recruit_on_cell"]) for r in rows),
                  enemy_recruited_onto_target=sum(bool(r["pay"]["enemy_spawned_at_target"]) for r in rows))
    return result


def strata(rows,f):
    groups=defaultdict(list)
    for r in rows: groups[f(r)].append(r)
    return {k:examined(v) for k,v in sorted(groups.items())}


def reach(r):
    return "reachable" if r["public_reach"]["existing"] else "unreachable"


def map_effects(rows):
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:groups[r["map"]][reach(r)].append(r)
    keep={m:q for m,q in groups.items() if set(q)=={"reachable","unreachable"}}
    effects={}
    for m,q in keep.items():
        effects[m]={}
        for k in ("survived","charged","parent_survived"):
            def outcome(r):
                if k=="survived":return float(r["pay"]["survived"])
                if k=="charged":return float(r["pay"]["worker_debit"]>0)
                return float(r["parent"]["survived"])
            effects[m][k]=mean(outcome(r) for r in q["reachable"])-mean(outcome(r) for r in q["unreachable"])
    mean_diffs={}
    if effects:
        rng=random.Random(20261119)
        maps=sorted(effects)
        resamples=[[rng.choice(maps) for _ in maps] for _ in range(5000)]
        for metric in next(iter(effects.values())):
            boot=sorted(mean(effects[m][metric] for m in draw) for draw in resamples)
            mean_diffs[metric]={"mean_within_map_reachable_minus_unreachable":mean(effects[m][metric] for m in maps),
                                "percentile_map_bootstrap_95":[boot[124],boot[4874]]}
    return {"eligible_mixed_maps":len(keep),"all_terrains":len(groups),"per_map":effects,"differences":mean_diffs,
            "interpretation":"Map-paired reachable minus unreachable, equal terrain weight; exploratory conditional interval. Maps without both classes do not identify within-map separation."}


def main():
    plan=read(HERE/"panel.json"); games=[];ids=[]
    for row in plan["games"]:
        p=HERE/"games"/f'{row["seed"]}_{row["seat"]}.json.gz'
        d=read(p)
        check(d["identity"]=={"panel_sha256":sha(HERE/"panel.json"),"runner_sha256":sha(HERE/"transfer.py")},"wrong evidence identity")
        g=d["game"]
        check((g["map"],g["seed"],g["seat"])==(row["map"],row["seed"],row["seat"]),"wrong game")
        games.append(g)
        ids.append({"name":p.name,"sha256":sha(p)})
    rows=[{**r,"seat":g["seat"],"map":g["map"]} for g in games for r in g["selected"]]; changed=[r for r in rows if r["changed"]]
    unchanged=[r for r in rows if not r["changed"]]
    def group(r):
        return reach(r)+"/"+("arrived" if r["pay"]["enemy_real_existing_arrivals"] else "stayed_away")
    separators={"all_selected":examined(rows),"changed":examined(changed),"already_paying":examined(unchanged),
       "all_selected_by_public_reach":strata(rows,reach),"changed_by_public_reach":strata(changed,reach),
       "changed_by_reach_and_hindsight":strata(changed,group),"already_paying_by_reach":strata(unchanged,reach),
       "changed_by_kind_and_reach":strata(changed,lambda r:r["kind"]+"/"+reach(r)),
       "changed_by_seat_and_reach":strata(changed,lambda r:str(r["seat"])+"/"+reach(r)),
       "changed_by_window_and_reach":strata(changed,lambda r:str(r["window"])+"/"+reach(r)),
       "changed_by_map_and_reach":strata(changed,lambda r:r["map"]+"/"+reach(r)),"map_contrasts":map_effects(changed)}
    out={"scope":"Independent from the original 16 pay maps only. These are historically used development maps with the exact same v1 parent and unmodified deployed opponent.",
         "panel_sha256":sha(HERE/"panel.json"),"runner_sha256":sha(HERE/"transfer.py"),"summarizer_sha256":sha(HERE/"summarize.py"),
         "evidence":ids,"denominators":{"terrains":len({r["map"] for r in games}),"map_seats":len(games),
                 "reached_windows":sum(g["available_windows"] for g in games),"registered_windows":2*len(games),
                 "chronological_prefix_steps_checked_official":sum(g["official_parent_prefix_steps"] for g in games),
                 "actual_eligible_package_calls_until_selection":sum(sum(g["chronological_inspected_until_selection"]) for g in games),
                 "all_window_ticks_available":sum(sum(g["available_turns_per_window"]) for g in games),"selected":len(rows),"changed":len(changed)},
         **separators}
    save(HERE/"summary.json",out)
    print(out["denominators"])
    print({k:{f:d[f] for f in ("roots","terrains","pay_survived","pay_charged","parent_survived","pay_expense","parent_survived_pay_died","parent_died_pay_survived")} for k,d in out["changed_by_public_reach"].items()})
    print({k:v["roots"] for k,v in out["changed_by_reach_and_hindsight"].items()},out["map_contrasts"]["eligible_mixed_maps"],out["map_contrasts"]["differences"])


if __name__=="__main__":
    main()
