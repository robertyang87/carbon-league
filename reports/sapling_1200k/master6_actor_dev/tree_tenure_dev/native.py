"""No new games; test an already-trained public purchase rule on archived responsive counterfactuals."""
from collections import defaultdict
from functools import lru_cache
import gzip,hashlib,json,math
from pathlib import Path
import sys
import numpy as np
import model
from dataset import FEATURES

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parent
BASE=ACTOR.parent
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE/"all_top5"))
from plant_timing_extract import distances,CHOICE
from turning_point_ledger import FastCarbon
PARENT="9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0"
ORIGINAL="65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"


def read(p):
    data=p.read_bytes()
    return json.loads(gzip.decompress(data) if p.suffix==".gz" else data)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(v,msg):
    if not v:raise ValueError(msg)


def features(view,cfg,uid):
    t,s=view["step"],view["player"]
    own,opp=view["players"][s],view["players"][1-s];n=cfg["size"];life=cfg["treeLifespan"]
    cell,cargo,job=own[2][uid]
    verify(job=="PLANTER" and cell not in set(own[1].values())|set(opp[1].values()),"nonpurchase intervention")
    trees={pos:(p,age) for p,row in enumerate(view["players"]) for pos,age in row[3].values()}
    alive=sum(age<life for p,age in trees.values());old=trees.get(cell)
    verify(not(old and old[0]==s and old[1]<life),"already owns living tree")
    kind="seize" if old and old[1]<life else "plant"
    price=cfg["seizeCost"] if kind=="seize" else cfg["plantCost"]
    if kind=="plant" and len(own[3])>cfg["playerTreeProtectiveNumber"]:
        price+=cfg["plantCostInflationRatio"]*cfg["plantCostInflationBase"]**alive
    y,x=divmod(cell,n);nbr=[(y-1)%n*n+x,y*n+(x+1)%n,(y+1)%n*n+x,y*n+(x-1)%n]
    dist=distances(n)[cell]
    def nearest(p,k):return min((dist[pos] for pos,_,role in p[2].values() if role==k),default=99)
    values=dict(kind=kind,price=round(price,4),neighbor_carbon=round(sum(view["carbon"][j] for j in nbr if j not in trees),2),
                cash_after_price=round(own[0]-price,3),remaining_turns=cfg["episodeSteps"]-1-t,
                old_tree_age=old[1] if old else -1,opp_planter_dist=nearest(opp,"PLANTER"),own_collector_dist=nearest(own,"COLLECTOR"),
                opp_collector_dist=nearest(opp,"COLLECTOR"),alt_adjacent_enemy_tree=sum(j in trees and trees[j][0]!=s and trees[j][1]<life-1 for j in nbr),
                own_trees=len(own[3]),alive_global_trees=alive,own_home_dist=min(dist[j] for j in own[1].values()),
                opp_home_dist=min(dist[j] for j in opp[1].values()))
    verify(set(FEATURES)<=set(values)<=set(CHOICE) and not ({"paid","command","next_same_worker_paid_in_5"}&set(values)),"feature leakage")
    return values,price


def mean(xs):return sum(xs)/len(xs) if xs else None


def rollup(rows,rule):
    maps=defaultdict(list)
    for r in rows:
        choice=r["rule"][rule];a=r["arms"][choice];p=r["arms"][r["parent_arm"]]
        maps[r["map"]].append({"own":a["own"]-p["own"],"minus_opp":p["opp"]-a["opp"],"margin":a["margin"]-p["margin"],
                               "absolute_margin":a["margin"],"absolute_own":a["own"],"absolute_opp":a["opp"],
                               "action_changed":int(choice!=r["parent_arm"]),"pay":int(choice=="pay"),"paid":int(a["debit"]>0),
                               "root_price_paid":a["debit"],"won":int(a["own"]>a["opp"])})
    metrics={k:mean([mean([t[k] for t in separate]) for separate in maps.values()]) for k in next(iter(maps.values()))[0]}
    dx=[mean([e["margin"] for e in group]) for group in maps.values()]
    check_sum=metrics["own"]+metrics["minus_opp"]-metrics["margin"];verify(abs(check_sum)<1e-5,"terminal decomposition")
    return {"map_equal":metrics,"roots":len(rows),"terrains":len(maps),
            "pay_orders":sum(t["pay"] for a in maps.values() for t in a),"actual_payments":sum(t["paid"] for a in maps.values() for t in a),
            "changed_from_v1":sum(t["action_changed"] for a in maps.values() for t in a),"by_terrain_mean_delta":dict(zip(maps,dx)),
            "positive_flat_negative_map_counts":[sum(d>1e-8 for d in dx),sum(abs(d)<=1e-8 for d in dx),sum(d< -1e-8 for d in dx)]}


def run():
    trained=model.load()
    package=ACTOR/"strong_teacher_dev/proposals.json.gz";q=read(package)
    rows=[];groups=defaultdict(dict)
    verify(q["identity"]["original_examiner_sha256"]==ORIGINAL and len(q["rows"])==64,"incorrect original panel")
    for r in q["rows"]:
        verify(r["t"] not in groups[r["source"]],"duplicate source root")
        groups[r["source"]][r["t"]]=r
    for src,when in groups.items():
        path=ROOT/src;raw=read(path);cfg=raw["game_configuration"];s=raw["candidate_seat"];seed=raw["seed"]
        verify(not raw["errors"] and raw["pack_sha256"]==PARENT and raw["examiner_md5"]=="57f9a594cce4f2ec9e3573d8d9d26ca1","parent/original mismatch")
        sim=FastCarbon(cfg).reset(seed=seed)
        for t in range(max(when)+1):
            if t not in when:
                sim.step(*raw["actions"][t]);continue
            r=when[t];wd=ROOT/r["case"];lp=read(wd/"live_protocol.json");root=read(wd/"root.json.gz");signed=read(wd/"training_label.json")
            view=sim.obs(s);a=raw["actions"][t]
            verify(r["source_sha256"]==sha(path) and r["seat"]==s and r["seed"]==seed and r["source_terrain_sha256"]==raw["initial_terrain_u8_sha256"],"source identity")
            verify(r["public_before_sha256"]==hashlib.sha256(json.dumps(view,sort_keys=True).encode()).hexdigest(),"strong public root")
            verify(all(view[k]==root["public_at_root"][k] for k in ("step","player","players","trees","carbon")),"live root observation")
            verify(signed["root_sha256"]==sha(wd/"root.json.gz"),"registered root")
            f,quoted=features(view,cfg,r["uid"])
            dur,gross=map(float,trained.predict([f])[0]);pay=gross>f["price"]
            verify(math.isfinite(dur) and math.isfinite(gross),"nonfinite forecast")
            parent_arm="pay" if signed["pay_is_v1"] else "defer"
            arms={}
            for key in ("pay","defer"):
                label=signed[key];nat=label["native_arm"];filename=wd/f"live_{nat}.json";e=read(filename);tape=wd/e["action_tape"]
                actual=read(tape)
                verify(e["source_sha256"]==ORIGINAL and e["pack_sha256"]==PARENT and e["protocol_sha256"]==sha(wd/"live_protocol.json")
                       and e["original_trajectory_sha256"]==r["source_sha256"] and sha(filename)==r["native"][nat]["source_sha256"]
                       and sha(tape)==e["action_sha256"]==label["actual_action_tape_sha256"]==r["native"][nat]["tape_sha256"]
                       and e["shared_init_sha256"]==sha(wd/"live_shared_init.json")==label["shared_stream_sha256"],"invalid native arm")
                verify(e["terminal_t"]==t+len(actual) and e["search_calls"]==len(actual)-1 and e["statuses"]==["DONE","DONE"]
                       and e["score"]==label["scores_by_physical_seat"]==r["native"][nat]["score"],"unverified original termination")
                verify((actual[0][s].get(r["uid"]) in (None,"NONE"))==(key=="pay") and actual[0][1-s]==a[1-s],"root decision conflict")
                verify(abs(quoted-label["quoted"])<.006 and (key=="pay" or label["root_debit"]==0),"incorrect actual charge")
                scored=e["score"];duration=None
                if label["root_debit"]>0:
                    duration=(label["first_tenure_ended_at"]-(t+1) if label["first_tenure_ended_at"] is not None else e["terminal_t"]-t)
                arms[key]={"terminal_physical_scores":scored,"own":scored[s],"opp":scored[1-s],"margin":e["margin"],
                           "debit":label["root_debit"],"worker_survived_first":label["root_worker_survived"],"duration":duration,
                           "origin_tree_id":label["originating_tree_id"],"first_tenure_receipts":label["first_tenure_income"],
                           "native_arm":nat,"native_proof_sha256":sha(filename),"action_sha256":sha(tape),"original_later_search_calls":e["search_calls"]}
            verify((arms["pay"]["debit"]>0)==signed["pay_intent_filled"] and arms[parent_arm]["native_arm"]=="parent","wrong pay mapping")
            rows.append({"seed":seed,"seat":s,"t":t,"uid":r["uid"],"split":r["split"],"map":r["source_terrain_sha256"],"case":r["case"],"kind":f["kind"],
                         "features":f,"source_sha256":r["source_sha256"],"public_before_sha256":r["public_before_sha256"],"official_label_sha256":sha(wd/"training_label.json"),
                         "forecast":{"duration":dur,"gross":gross,"exact_quoted":quoted},"arms":arms,"parent_arm":parent_arm,
                         "rule":{"model":"pay" if pay else "defer","parent":parent_arm,"always_pay":"pay","always_defer":"defer"}})
            sim.step(*a)
    verify(len(rows)==64 and sum(r["split"]=="hold" for r in rows)==16 and len({r["map"] for r in rows})==16,"incomplete split")
    verify({r["map"] for r in rows if r["split"]=="train"}.isdisjoint({r["map"] for r in rows if r["split"]=="hold"}),"split overlap")
    summary={}
    for split in ("all","train","hold"):
        sub=[r for r in rows if r["split"]==split or split=="all"]
        buy=[r for r in sub if r["arms"]["pay"]["debit"]>0]
        summary[split]={"comparison":{k:rollup(sub,k) for k in rows[0]["rule"]},"actual_pay_orders_filled":len(buy),
                        "model_diagnostics_actual_buy_filled":{"duration_mae":mean([abs(r["forecast"]["duration"]-r["arms"]["pay"]["duration"]) for r in buy]),
                        "gross_mae":mean([abs(r["forecast"]["gross"]-r["arms"]["pay"]["first_tenure_receipts"]) for r in buy])}}
    diffs=[]
    for r in rows:
        a=r["arms"][r["rule"]["model"]];p=r["arms"][r["parent_arm"]]
        if r["rule"]["model"]!=r["parent_arm"]:
            diffs.append({"case":r["case"],"split":r["split"],"seed":r["seed"],"seat":r["seat"],"t":r["t"],"uid":r["uid"],"kind":r["kind"],
                          "model_decision":r["rule"]["model"],"parent_decision":r["parent_arm"],"own_score_delta":a["own"]-p["own"],
                          "minus_opp_score_delta":p["opp"]-a["opp"],"delta_margin":a["margin"]-p["margin"]})
    diffs.sort(key=lambda v:v["delta_margin"])
    return {"protocol":"model buys iff predicted own gross exceeds CHOICE price; one historical action change, then same original v1 continuation with authentic opponent at temperature one",
            "source_comparison":"Model independent of native labels. Train/hold are historical native labels only; all 64 roots are previous development data. No complete new games.",
            "source_sha256":sha(package),"model_sha256":sha(HERE/"model.py"),"weights_sha256":sha(HERE/"weights.npz"),"code_sha256":sha(Path(__file__)),
            "features_in_order":FEATURES,"rule_equivalent_using_exact_quote":all((r["forecast"]["gross"]>r["forecast"]["exact_quoted"])==(r["rule"]["model"]=="pay") for r in rows),
            "summary":summary,"changed_cases_sorted_by_margin":diffs,"rows":rows}


if __name__=="__main__":
    doc=run()
    (HERE/"native.json.gz").write_bytes(gzip.compress(json.dumps(doc,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode(),mtime=0))
    for k,d in doc["summary"].items():
        x=d["comparison"]["model"];print(k,x["terrains"],x["roots"],x["pay_orders"],x["actual_payments"],x["changed_from_v1"],x["map_equal"]["margin"])
