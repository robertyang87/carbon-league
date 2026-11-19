"""Census of first actual candidate base substitutions on six real complete v1 games."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
from itertools import permutations
import json
import sys
from build import BASE,DEV,HERE,ROOT,SOURCE,read,write,sha,emit

sys.path[:0]=[str(BASE),str(ROOT/"code")]
from fastsim import FastCarbon
from map_identity import terrain_sha256


def one(path):
    raw=read(path);who=raw["candidate_seat"];seed=raw["seed"];cfg=raw["game_configuration"]
    meta=read(path.parent/"official_replay.json")["runs"][path.parent.name]
    found=[u for u in meta["games"] if (u["seed"],u["seat"])==(seed,who)];assert len(found)==1
    assert sha(path)==found[0]["transcript_sha256"] and raw["pack_sha256"]==sha(SOURCE) and found[0]["statuses"]==["DONE","DONE"]
    spec=importlib.util.spec_from_file_location("workforce_frozen_screen",HERE/"agent.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    sim=FastCarbon(cfg).reset(seed=seed);assert terrain_sha256(sim.carbon)==raw["initial_terrain_u8_sha256"]
    reasons=Counter();chosen=None;root_proposals=0
    for t,acts in enumerate(raw["actions"]):
        obs=sim.obs(who)
        decision=mod.agent(obs,cfg);info=mod._WF_LAST
        reasons[info["reason"]]+=1
        if info["reason"] in ("committed","no_safe_confident_change"):root_proposals+=1
        old=acts[who];changed={uid for uid in (old.keys()|decision.keys()) if old.get(uid)!=decision.get(uid)}
        if changed:
            assert info["changed"] and chosen is None and changed=={info["base"]}
            assert set(changed)<=obs["players"][who][1].keys()
            chosen=dict(source=str(path.relative_to(ROOT)),source_sha256=sha(path),seed=seed,seat=who,t=t,base=info["base"],
                        parent_joint=old,alternative_joint=decision,root_opponent_fixed=acts[1-who],
                        paired_rng_seed=int.from_bytes(hashlib.sha256(f"workforce-first-v1|{seed}|{who}|{t}".encode()).digest()[:4],"big"),
                        trigger=info,own_cash=obs["players"][who][0],preexisting_workers=obs["players"][who][2],
                        initial_terrain_u8_sha256=raw["initial_terrain_u8_sha256"])
        else:
            assert not info["changed"]
        sim.step(*acts)
    return dict(seed=seed,seat=who,turns=len(raw["actions"]),source_sha256=sha(path),source=str(path.relative_to(ROOT)),
                reason_counts=dict(reasons),first=chosen,eligible_parent_turns_until_commit=root_proposals)


def observationally_eligible():
    evidence=read(HERE/"observational_gate.json");ledger=read(HERE/"baseline_summary.json")
    assert evidence["baseline_summary_sha256"]==sha(HERE/"baseline_summary.json")
    assert evidence["baseline_games_sha256"]==ledger["evidence_sha256"]==sha(HERE/"baseline_games.json.gz")
    values={s:ledger["sides"][s]["already_two_planters_pre220"]["P"] for s in ("v1","Master3s2")}
    assert evidence["official_reference_counts"]==values
    a,b=values["v1"],values["Master3s2"]
    eligible=(a["hires"]>b["hires"] and a["direct_lifetime_positive"]/a["hires"]<b["direct_lifetime_positive"]/b["hires"])
    assert evidence["route_eligible"]==eligible
    return eligible


def main():
    opportunity=observationally_eligible()
    emit();evaluation=read(HERE/"evaluation.json")
    paths=sorted((DEV/"terminal_return_fit/replication/learned").glob("raw_*.json.gz"))+sorted((DEV/"terminal_return_fit/dev_learned").glob("raw_*.json.gz"))
    assert len(paths)==6
    with ProcessPoolExecutor(max_workers=2) as p:rows=list(p.map(one,paths))
    cases=[r["first"] for r in rows if r["first"] is not None]
    opts=[(a,b) for a,b in permutations(cases,2) if a["seat"]==0 and b["seat"]==1 and a["initial_terrain_u8_sha256"]!=b["initial_terrain_u8_sha256"]]
    # Ordered map pool from the declared pair; second map paired first when both are available.
    order=list(dict.fromkeys(r["seed"] for r in rows))
    opts.sort(key=lambda z:(order.index(z[0]["seed"]),order.index(z[1]["seed"])))
    selected=list(opts[0]) if opportunity and evaluation["passed"] and opts else []
    write(HERE/"screen.json",dict(model_evaluation_passed=evaluation["passed"],games=rows,eligible_first_roots=len(cases),
         distinct_maps_with_first=len({r["seed"] for r in cases}),alternative_roots_eligible_for_two_map_pair=len(opts)))
    write(HERE/"roots.json",dict(parent_sha256=sha(SOURCE),candidate_sha256=sha(HERE/"agent.py"),screen_sha256=sha(HERE/"screen.json"),
                               roots=selected,stage="approved two conditional branches" if selected else "no native branches"));
    print("actual first commits",[(r["seed"],r["seat"],r["t"]) for r in cases],"frozen",[(r["seed"],r["seat"],r["t"]) for r in selected],flush=True)


if __name__=="__main__":main()
