"""Construct rule counterexamples and replay real seat-specific observations through both deploy switches."""
from collections import Counter
from copy import deepcopy
import gzip
import json
from pathlib import Path
from types import ModuleType

from pydantic_core import from_json

import actions
import build

HERE=build.HERE
CACHE=Path("/workspace/sapling-all-top5-20261119/replays")
PANEL=HERE.parent/"late_advantage_actor_dev/panel.json"
DEFAULT={"size":15,"episodeSteps":300,"recCollectorCost":30,"recPlanterCost":30,
         "recCollectorIncreaseCost":0,"recPlanterIncreaseCost":0,"workerLimit":10}


def check(condition,case):
    if not condition:
        raise ValueError(case)


def mechanical():
    fixtures=[]
    for seat in (0,1):
        def state(cargo=100.,enemies=None,workers=None,bases=None):
            ours=[200.,bases or {"home":0},{"c":[34,cargo,"COLLECTOR"],**(workers or {})},{}]
            them=[200.,{"o":198},enemies if enemies is not None else {"e":[32,0.,"COLLECTOR"]},{}]
            return {"step":180,"player":seat,"players":[ours,them] if seat==0 else [them,ours],
                    "trees":{},"carbon":[0.]*225}
        def evaluated(label,obs,extra=None,commands=None):
            score={"LEFT":5.,"UP":4.,"RIGHT":3.,"DOWN":2.,"NONE":1.}
            rows=actions.ranked_options(obs,DEFAULT,{"c":"LEFT",**(commands or {})},{"c":score})
            passed=[r for r in rows if r["threat"]["strict"]]
            fixtures.append({"seat":seat,"case":label,"offered":[r["action"] for r in rows],"strict":[r["action"] for r in passed]})
            return rows,passed
        _,r=evaluated("loaded_80_or_more",state(80.));check(r and r[0]["action"]=="UP" and r[0]["parent_rank"]==2,"safe second action")
        _,r=evaluated("cargo_just_under",state(79.999));check(not r,"cargo threshold")
        _,r=evaluated("empty_can_take_heavier_enemy",state(0,{"e":[32,105,"COLLECTOR"]}));check(not r,"pursuit")
        _,r=evaluated("loaded_can_take_heavier_enemy",state(100,{"e":[32,101,"COLLECTOR"]}));check(not r,"heavier only")
        offer,r=evaluated("equal_only_never_activate",state(100,{"e":[32,100,"COLLECTOR"]}));check(offer and not r,"equal only")
        _,r=evaluated("optional_second_lighter_same_destination",state(100,{"a":[32,0,"COLLECTOR"],"b":[18,0,"COLLECTOR"]}))
        check(r and r[0]["action"]=="RIGHT" and len(r[0]["threat"]["strict"])==2 and
              r[0]["threat"]["minimum_tied_if_all_arrive"],"overlap does not imply safe")
        _,r=evaluated("strict_and_equal_enemy_may_individually_commit",state(100,{"a":[32,0,"COLLECTOR"],"b":[18,100,"COLLECTOR"]}))
        check(r and r[0]["threat"]["equal"]==["b"] and r[0]["threat"]["strict"]==["a"],"optional tie")
        _,r=evaluated("avoid_friendly_stop",state(workers={"p":[19,0,"PLANTER"]}))
        check(r and all(p["destination"]!=19 for p in r),"friendly occupied")
        _,r=evaluated("avoid_hiring_base",state(bases={"home":0,"second":19}),commands={"second":"RECCOLLECTOR"})
        check(r and all(p["destination"]!=19 for p in r),"recruit destination")
        _,r=evaluated("do_not_block_last_route_of_heavier_teammate",
                      state(workers={"high":[20,250,"COLLECTOR"],"p":[5,0,"PLANTER"]}))
        check(r and all(p["destination"]!=19 for p in r),"team opportunity")
        _,r=evaluated("no_safe_action_no_edit",state(enemies={str(k):[k,0,"COLLECTOR"] for k in (19,33,35,49)}))
        check(not r,"no safe improvement")
        _,r=evaluated("cannot_deliver_before_clock",dict(state(),step=298))
        check(not r,"out of time")
    return fixtures


def module(raw,name):
    m=ModuleType(name)
    exec(compile(raw,name,"exec"),vars(m))
    return m


def real():
    parent_source=build.PARENT.read_bytes()
    enabled_source,enabled=build.compose()
    disabled_source,disabled=build.compose(False)
    check(enabled_source==(HERE/"candidate.py").read_bytes(),"current shipped package")
    rows=json.loads(PANEL.read_bytes())
    selected=[]
    for s in (0,1):
        selected.extend(sorted((r for r in rows if r["split"]=="held" and r["seat"]==s and r["turns"]>=299),key=lambda r:r["id"])[:2])
    check(len({r["terrain_sha256"] for r in selected})==4,"four different already-exposed development maps")
    reports=[]
    all_counts=Counter()
    for card in selected:
        replays=(CACHE/(card["id"]+".json.gz")).read_bytes()
        check(build.sha(replays)==card["sha256"],"source replay hash")
        game=from_json(gzip.decompress(replays))
        p=module(parent_source,"pure_parent")
        scored=module(parent_source,"instrumented_parent")
        on=module(enabled_source,"enabled_candidate")
        off=module(disabled_source,"disabled_candidate")
        actions.install_parent_trace(scored,parent_source,on._local_edit)
        seat=card["seat"];cfg={**on._CARGO_CFG,**(game.get("configuration") or {})}
        turns=[];used=False;count=Counter();coverage=Counter()
        for t,frame in enumerate(game["steps"][:299]):
            view={k:frame[0]["observation"][k] for k in ("step","players","trees")}
            view.update(player=seat,carbon=frame[seat]["observation"]["carbon"])
            if t not in (0,179,296,297,298) and not 180<=t<=295:
                for actor in (p,scored,on,off):
                    actor._FEAT.observe(view)
                continue
            baseline=p.agent(view,cfg)
            observed=scored.agent(view,cfg)
            poison={**view,"reveal":{"14":999999.},"full_carbon":[99999.]*225}
            actual=on.agent(poison,cfg);disabled_order=off.agent(poison,cfg)
            check(baseline==observed==disabled_order==on._CARGO_TRACE["parent"]==off._CARGO_TRACE["parent"],("parent parity",t))
            changed={u for u in actual.keys()|baseline.keys() if actual.get(u)!=baseline.get(u)}
            stage=180<=t<=295
            options=actions.candidates(view,cfg,observed,scored) if stage else []
            best=next((a for a in options if a["threat"]["strict"]),None)
            if best is not None:
                coverage["strict_safe_roots"]+=1
            if any(a["threat"]["equal"] and not a["threat"]["strict"] for a in options):
                coverage["equality_only_units_available"]+=1
            if any(len(a["threat"]["commitments"])>=2 for a in options):
                coverage["multiple_enemy_roots"]+=1
            if stage:
                reach=actions.common_enemy_reach(view,cfg,15)
                for uid,(start,cargo,role) in view["players"][seat][2].items():
                    if role!="COLLECTOR" or cargo<actions.MIN_CARGO:
                        continue
                    move=baseline.get(uid,"NONE");dest=actions.pos_after(start,move,15)
                    amount=cargo if move=="NONE" else cargo*(1-cfg.get("moveCost",0))
                    if actions.exposure(reach,dest,amount)["strict"]:
                        coverage["parent_high_cargo_strictly_threatened_units"]+=1
            expected=best if not used else None
            if expected:
                checked=on._CARGO_TRACE["chosen"]
                check(changed=={expected["uid"]} and checked==expected and actual.get(expected["uid"],"NONE")==expected["action"],("real intervention",t))
                used=True
            else:
                check(not changed and on._CARGO_TRACE["chosen"] is None,("unintended edit",t))
            count[on._CARGO_TRACE["reason"]]+=1
            detail=[t,on._CARGO_TRACE["reason"],best,sorted(changed)]
            turns.append(detail)
        check(count["accepted"]<=1 and p._COUNTS==scored._COUNTS==on._COUNTS==off._COUNTS,"real parent stats and intervention budget")
        reports.append({"replay":card["id"],"source_sha256":card["sha256"],"seat":seat,"terrain":card["terrain_sha256"],
                        "public_turns":turns,"counts":dict(count),"opportunities":dict(coverage),"parent_network_contribution":deepcopy(p._COUNTS)})
        all_counts.update(count)
    return {"candidate_sha256":enabled["sha256"],"disabled_sha256":disabled["sha256"],"maps":reports,"counts":dict(all_counts)}


def main():
    tests=mechanical()
    observed=real()
    result={"fixture_sha256":build.sha(Path(__file__).read_bytes()),"actions_sha256":build.sha((HERE/"actions.py").read_bytes()),
            "selection":"Two lexicographically smallest old development replay IDs per seat with 299 decisions; outcomes unused.",
            "related_prior":{"single_tile_possible_loss":"reports/master_family/tactical_rules/defense.py",
              "late_learner":"reports/sapling_1200k/master6_actor_dev/stratified_action_value_dev/actions.py, evade category",
              "attacking_rival_freight":"reports/sapling_1200k/master6_actor_dev/freight_response/README.md",
              "external_baselines":"reports/GEMINI_APEX_V1FIXED_20261119.md and reports/GEMINI_APEX_V1FIX2_20261119.md",
              "difference":"New code uses post-guard combined Sapling action scores, separate strictly lighter/tied alternative arrivals, high-capital and teammate-corridor veto. No external policy weights, code or logits used."},
            "controls":tests,"serving":observed}
    build.write(HERE/"action_fixture.json.gz",gzip.compress((json.dumps(result,separators=(",",":"),allow_nan=False)+"\n").encode(),mtime=0))
    print(observed["counts"],[(m["seat"],m["opportunities"]) for m in observed["maps"]])


if __name__=="__main__":
    main()
