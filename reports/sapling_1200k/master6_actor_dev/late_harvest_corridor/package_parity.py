"""Verify actual single-file neural decisions against the signed v1 and independent mission rule."""
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import data
import policy

HERE=Path(__file__).resolve().parent
PACK="306f0bd1661975840a7c92911b08f2ce324b0f16bda678818068778d5a41e1ac"


def clean(x):return {k:v for k,v in x.items() if v and v!="NONE"}
def normalize(x):return json.loads(json.dumps(x,sort_keys=True,allow_nan=False))
def same(a,b,why):data.must(normalize(a)==normalize(b),why)
def changes(a,b):return {u:[a.get(u,"NONE"),b.get(u,"NONE")] for u in sorted(a.keys()|b.keys()) if a.get(u,"NONE")!=b.get(u,"NONE")}


def load_candidate():
    candidate=HERE/"candidate.py";meta=data.read(HERE/"package.json")
    data.must(data.sha(candidate)==meta["candidate_sha256"]==PACK and data.sha(HERE/"policy.py")==meta["rules_sha256"] and
              data.sha(HERE/"protocol.json")==meta["protocol_sha256"],"package identity")
    parent=(data.ROOT/"data/weights/sapling_1200k/terminal_return_fit_v1.py").read_bytes()
    data.must(hashlib.sha256(parent).hexdigest()==meta["parent_sha256"] and candidate.read_bytes().startswith(parent),
              "v1 predecessor bytes")
    spec=importlib.util.spec_from_file_location("actual_late_harvest_candidate",candidate)
    pack=importlib.util.module_from_spec(spec);spec.loader.exec_module(pack)
    return pack


def cancel_fixture(pack,g,t,mission):
    obs=deepcopy(g["observed"][t+1]);uid=mission["worker"];s=g["seat"]
    obs["players"][s][2].pop(uid,None)
    saved = {k:deepcopy(getattr(pack,k)) for k in ("_FEAT","_COUNTS","_HARVEST_PLAN","_HARVEST_HISTORY",
                  "_HARVEST_ISSUED","_HARVEST_LAST","_HARVEST_TRACE")}
    try:
        output=clean(pack.agent(obs,g["cfg"]))
        _,_,base,telemetry=pack._HARVEST_TRACE
        exp,new,meta=policy.choose(obs,{t:deepcopy(g["observed"][t])},dict(base),g["cfg"],deepcopy(mission),False)
        same(output,clean(exp),"synthetic actual command");same(telemetry,meta,"synthetic trace")
        data.must(meta["status"]=="abort" and meta["stage"]=="lost_worker_or_history" and new is None
                  and pack._HARVEST_PLAN is None and pack._HARVEST_ISSUED and output==clean(base),"unsafe worker loss")
        return dict(seed=g["seed"],seat=s,t=t+1,removed_uid=uid,status=meta["status"],synthesized=True)
    finally:
        for k,value in saved.items():setattr(pack,k,value)


def main():
    pack=load_candidate();lengths=Counter();tally=Counter();reports=[];synth=None
    receipts=data.read(HERE/"source_registry.json")
    identities={(r["seed"],r["seat"]):r for r in receipts["games"]}
    for g in data.load_public()["games"]:
        seed,s=g["seed"],g["seat"];record=identities[(seed,s)]
        rpath=data.ROOT/record["path"];data.must(data.sha(rpath)==record["raw_sha256"],"source changed")
        raw=data.read(rpath);timeline=g["observed"];cfg=g["cfg"]
        # When testing the next game, pre-existing wrapper memory really is dirty.
        plan=None;issued=False;standalone_first=None;actual_first=None;rows=[]
        for t,obs in enumerate(timeline):
            if 0<t<g["first"]:
                pack._FEAT.observe(obs)
                pack._harvest_remember(obs)
                pack._HARVEST_LAST=(s,t)
                continue
            past={u:timeline[u] for u in range(max(0,t-12),t)}
            logged=clean(raw["actions"][t][s])
            exp,plan_next,meta=policy.choose(obs,past,dict(logged),cfg,deepcopy(plan),allow_start=not issued)
            exp=clean(exp);stamp=time.monotonic()
            have=clean(pack.agent(obs,cfg));elapsed=time.monotonic()-stamp
            k,p,actual_base,actual_meta=pack._HARVEST_TRACE
            data.must((k,p)==(t,s),"stale trace")
            same(clean(actual_base),logged,f"actual v1 diverges: {seed}/{s}/{t}")
            same(have,exp,"emitted joint command");same(actual_meta,meta,"status/option metadata")
            same(pack._HARVEST_PLAN,plan_next,"mission state")
            same(pack._HARVEST_ISSUED,issued or meta["status"]=="start","one-mission latch")
            data.must(set(pack._HARVEST_HISTORY)==set(range(max(0,t-1),t+1)),"retained stale or future history")
            if t==0:
                data.must(meta["status"]=="fallback" and pack._HARVEST_PLAN is None and not pack._HARVEST_ISSUED,"opening reset")
                tally["opening_real_calls"]+=1
            else:
                free,_,free_meta=policy.choose(obs,past,dict(logged),cfg)
                if standalone_first is None and clean(free)!=logged:standalone_first=t
                if actual_first is None and have!=logged:actual_first=t
                tally["live_endgame_calls"]+=1;tally[meta["status"]]+=1
                tally["deviation_ticks"]+=have!=logged
                rows.append({"t":t,"base":logged,"emitted":have,"changes":changes(logged,have),
                             "decision":meta,"next_mission":plan_next,"first_eligible_if_still_unissued":free_meta["status"]=="start",
                             "selection_inputs":"focal observation, public past, own baseline", "observed_local_seconds":elapsed})
                if synth is None and plan_next is not None and t+1<len(timeline):
                    synth=cancel_fixture(pack,g,t,plan_next)
            plan=plan_next
            issued|=meta["status"]=="start"
        data.must(standalone_first==actual_first,"first difference from independent scan")
        counts=Counter(v["decision"]["status"] for v in rows)
        lengths[g["split"]]+=actual_first is not None
        reports.append({"seed":seed,"seat":s,"split":g["split"],"raw_sha256":record["raw_sha256"],"first_actual_change":actual_first,
                        "first_standalone_change":standalone_first,"status_counts":counts,"prewindow_visible_history_updates":g["first"]-1,
                        "rows":rows})
        print(seed,s,actual_first,dict(counts),flush=True)
    data.must(tally["opening_real_calls"]==16 and tally["live_endgame_calls"]==192,"incomplete runway")
    data.reference().write(HERE/"package_parity.json.gz",{"candidate_sha256":PACK,"implementation_sha256":data.sha(Path(__file__)),
            "data_receipt_sha256":data.sha(HERE/"data_receipt.json"),"package_receipt_sha256":data.sha(HERE/"package.json"),
            "counts":dict(tally),"first_eligible_games_by_split":dict(lengths),"synthetic_worker_loss":synth,
            "scope":"prefix matched v1 commands on real trajectories; clocks are shared low-priority CPU, not match latency; subsequent observations follow original recorded games",
            "games":reports})


if __name__=="__main__":
    main()
