"""Counterfactual single tree decisions on historical, full-Master3s2 development trajectories."""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

OUT=Path(__file__).resolve().parent
HERE=OUT.parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from official_game_replay import make
from all_top5.economy_extract import distances

PLAN=OUT/"protocol.json"
PACK=ROOT/"data/weights/sapling_1200k/hybrid_it50_v3.py"
PACK_HASH="b54cc51f4109f19f4a7b508eafc9cf04a28b7e295a9f442c5208f0d33f10a9ca"
DIRS={621454920:HERE/"terminal_economy/original_parent",621455920:HERE/"all_top5_pressure_dev/parent"}
MOVES={"NONE":(0,0),"UP":(-1,0),"RIGHT":(0,1),"DOWN":(1,0),"LEFT":(0,-1)}


def digest(data):return hashlib.sha256(data).hexdigest()


def destination(p,move,n):
    dr,dc=MOVES[move]
    return ((p//n+dr)%n)*n+(p%n+dc)%n


def clean(action):return {k:v for k,v in action.items() if v and v!="NONE"}


def utility(scores,seat):
    margin=scores[seat]-scores[1-seat]
    wdl=1. if margin>0 else 0. if margin<0 else .5
    return .5*wdl+.5/(1+math.exp(-max(-50.,min(50.,margin/1000))))


def begin(raw):
    np.random.seed(raw["seed"])
    env=make("carbon",configuration={**raw["game_configuration"],"randomSeed":raw["seed"]})
    if digest(bytes(env.states[0].observation.full_carbon))!=raw["initial_terrain_u8_sha256"]:
        raise ValueError("map identity")
    return env


def public(env,s):
    a=env.states[0].observation;b=env.states[s].observation
    return dict(player=s,step=a.step,players=a.players,trees=a.trees,carbon=b.carbon,remainingOverageTime=b.remainingOverageTime)


def candidate_roots(obs,raw,action):
    s=raw["candidate_seat"];t=obs["step"];n=raw["game_configuration"]["size"];dist=distances(n)
    cash,bases,ours,own_trees=obs["players"][s]
    _,enbases,theirs,op_trees=obs["players"][1-s]
    own_dest={uid:destination(v[0],action.get(uid,"NONE"),n) for uid,v in ours.items()}
    newcells={b for bid,b in bases.items() if action.get(bid) in ("RECCOLLECTOR","RECPLANTER")}
    enemy_can_damage=lambda pos,cargo,collector: any(dist[pos][w[0]]<=1 and
        (not collector or w[2]=="COLLECTOR" and w[1]<=cargo) for w in theirs.values())
    found=[]
    for uid,(cell,load,role) in ours.items():
        old=action.get(uid,"NONE")
        for tid,(target,age) in (own_trees.items() if role=="COLLECTOR" else op_trees.items()):
            if age>=raw["game_configuration"]["treeLifespan"] or target in enbases.values():continue
            if newcells & {target} or any(d==target for who,d in own_dest.items() if who!=uid):continue
            if own_dest[uid]==target:continue
            alt=next((k for k in MOVES if destination(cell,k,n)==target),None)
            if alt is None:continue
            if role=="PLANTER":
                if enemy_can_damage(target,load,False):continue
                if target==cell and cash>=raw["game_configuration"]["seizeCost"]:kind="capture_now"
                elif target!=cell and age<=35:kind="approach_young_tree"
                else:continue
            else:
                if age>35 or enemy_can_damage(target,load,True):continue
                if not any(w[2]=="PLANTER" and dist[target][w[0]]<=1 for w in theirs.values()):continue
                kind="defend_owned_tree"
            ident=f"tree-recovery-1|{raw['seed']}|{s}|{t}|{uid}|{tid}"
            found.append(dict(family=kind,t=t,worker=uid,origin=cell,cargo=load,old=old,alternative=alt,tree=tid,cell=target,
                              age=age,cash=cash,opponent_planter_dist=min((dist[target][w[0]] for w in theirs.values() if w[2]=="PLANTER"),default=None),
                              signature=digest(ident.encode())))
    return found


def source(raw,mod):
    env=begin(raw);seat=raw["candidate_seat"];selected={};counts=Counter();history=[]
    for t,commands in enumerate(raw["actions"]):
        obs=public(env,seat);history.append(obs)
        current=clean(mod.agent(obs,raw["game_configuration"]));recorded=clean(commands[seat])
        if current!=recorded:raise ValueError(f"parent action mismatch at {t}/{seat}: {current} vs {recorded}")
        if 100<=t<200:
            for r in candidate_roots(obs,raw,recorded):
                k=r["family"];counts[k]+=1
                if k not in selected or r["signature"]<selected[k]["signature"]:selected[k]=r
        env.step(commands)
    final=[x.reward for x in env.states]
    if not env.done or final!=raw["official_score_by_seat"]:raise ValueError("parent terminal mismatch")
    return history,selected,dict(counts),final


def counterfactual(raw,mod,history,root):
    seat=raw["candidate_seat"];env=begin(raw);root_t=root["t"];path=[]
    cfg=raw["game_configuration"];ts=Counter();not_present=0;stale_decisions=0;opponent_after=0
    for commands in raw["actions"][:root_t]:env.step(commands)
    # Observe and carry the full parent's historical feature tracker. Root moves
    # differ only after the baseline policy state and its score have been checked.
    feat=sys.modules["cnet2_feat"].Feat(seat)
    for obs in history[:root_t]:feat.observe(obs)
    mod._FEAT=feat
    d=0;t=root_t
    while not env.done:
        if t>=len(raw["actions"]):raise ValueError("recorded opponent tape exhausted before terminal")
        obs=public(env,seat);both=obs["players"];oldtrees=[p[3] for p in both]
        if t==root_t:
            alternative=clean(mod.agent(obs,cfg))
            if alternative!=clean(raw["actions"][t][seat]):raise ValueError("root policy failed reproduction")
            if root["alternative"]=="NONE":alternative.pop(root["worker"],None)
            else:alternative[root["worker"]]=root["alternative"]
        else:
            alternative=clean(mod.agent(obs,cfg))
        a=raw["actions"][t].copy();a[seat]=alternative
        op=1-seat
        present=set(both[op][1])|set(both[op][2])
        unknown=set(clean(a[op]))-present
        if t>root_t:
            not_present+=len(unknown);stale_decisions+=bool(unknown);opponent_after+=len(clean(a[op]))
            d+=dict(history[t]["players"][op][2])!=dict(both[op][2])
        env.step(a);t+=1;after=env.states[0].observation;loc=None
        for side in (seat,1-seat):
            if root["tree"] in after.players[side][3]:loc=side
        ts["owned_ticks"]+=loc==seat
        ts["tree_actual_income"]+=(after.trees.get(root["tree"],[None,0])[1] if loc==seat else 0)
        if t==root_t+1:first_owner=loc
        path.append(a)
    scores=[v.reward for v in env.states];statuses=[v.status for v in env.states]
    if statuses!=["DONE","DONE"]:raise ValueError(statuses)
    base=raw["official_score_by_seat"]
    return dict(root=root,terminal_t=t,terminal_status=statuses,official_scores=scores,margin_delta=round((scores[seat]-scores[1-seat])-(base[seat]-base[1-seat]),4),
                delta_u=utility(scores,seat)-utility(base,seat),first_owner=first_owner,
                target_tree_owned_ticks=ts["owned_ticks"],target_tree_income=round(ts["tree_actual_income"],4),
                opponent_worker_state_diff_turns=d,unavailable_opponent_commands=not_present,opponent_stale_id_turns=stale_decisions,
                opponent_commands_after_root=opponent_after,actions=path)


def main():
    if digest(PACK.read_bytes())!=PACK_HASH:raise ValueError("wrong model")
    spec=importlib.util.spec_from_file_location("sapling_recovery_parent",PACK);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    plan=json.loads(PLAN.read_bytes());resume=OUT/"observations";resume.mkdir(exist_ok=True)
    summary=[]
    for seed in plan["seeds"]:
        src=DIRS[seed];result=json.loads((src/"results.json").read_bytes())
        for seat in plan["seats"]:
            row=next(g for g in result["games"] if g["seed"]==seed and g["candidate_seat"]==seat)
            p=src/row["raw_file"];b=p.read_bytes()
            if digest(b)!=row["raw_sha256"]:raise ValueError("source mismatch")
            raw=json.loads(gzip.decompress(b))
            if raw["errors"] or raw["pack_sha256"]!=PACK_HASH or raw["examiner_md5"]!="57f9a594cce4f2ec9e3573d8d9d26ca1":raise ValueError("unexpected source provenance")
            cert=json.loads((src/"official_replay.json").read_bytes())["runs"][src.name]
            game_cert=next(v for v in cert["games"] if (v["seed"],v["seat"])==(seed,seat))
            if game_cert["transcript_sha256"]!=digest(b):raise ValueError("invalid official certificate")
            if raw.get("initial_terrain_u8_sha256",game_cert["map_sha256"])!=game_cert["map_sha256"]:raise ValueError("map identity disagreement")
            raw["initial_terrain_u8_sha256"]=game_cert["map_sha256"]
            hist,selected,n,score=source(raw,mod)
            group=dict(seed=seed,seat=seat,baseline_score=score,baseline_action_checks=len(hist),source=str(p.relative_to(ROOT)),
                       source_sha256=digest(b),eligible=n,roots={})
            for kind,root in selected.items():
                item=counterfactual(raw,mod,hist,root);actions=item.pop("actions")
                file=resume/f"{seed}-{seat}-{kind}.json.gz"
                payload=json.dumps(actions,separators=(",",":"),ensure_ascii=False).encode()
                file.write_bytes(gzip.compress(payload,mtime=0))
                item.update(branch_tape=str(file.relative_to(OUT)),branch_tape_sha256=digest(file.read_bytes()))
                group["roots"][kind]=item
                print(seed,seat,kind,root["t"],item["margin_delta"],round(item["delta_u"],5),flush=True)
            summary.append(group)
            (OUT/"results.json").write_text(json.dumps(dict(complete=len(summary)==4,protocol_sha256=digest(PLAN.read_bytes()),pack=PACK_HASH,games=summary),
                                                     ensure_ascii=False,allow_nan=False,indent=2)+"\n")


if __name__=="__main__":main()
