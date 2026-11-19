"""Score V5 final-step policy changes with the unmodified official board interpreter."""
import argparse
import base64
from collections import Counter
import copy
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zlib

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"code"))
sys.path.insert(0,str(ROOT/"code/env_src/carbon_challenge-master"))
import cnet2_feat as F
from cnet2_np import Policy
from build_sapling_cash_dagger import module
from fastsim import FastCarbon
from materialize_top5_h2h import read_repo,ARCHIVE,IDENTITIES
from zerosum_env import utils
from zerosum_env.envs.carbon.carbon import compute_next_board
from zerosum_env.envs.carbon import idgen


def judge(full,cfg,actions):
    if full["step"]!=cfg["episodeSteps"]-2:
        raise ValueError("expected last playable root")
    # Final cash and elimination are invariant to the unassigned suffix for a
    # newborn entity. Reserve a disjoint range to avoid overwriting historical IDs.
    idgen.reset()
    for seat in (0,1):
        for kind in ("tree-","worker-"):
            idgen._id_dict[f"player-{seat}-{kind}"]=10**6
    states=utils.structify([dict(status="ACTIVE",reward=full["players"][s][0],
                    observation=copy.deepcopy(full) if s==0 else {"player":1},
                    action=actions[s]) for s in (0,1)])
    compute_next_board(states,SimpleNamespace(configuration=utils.structify(cfg)))
    return [v.reward for v in states]


def compare(pack,pol,feat,x,obs,full,cfg,joint,rewards,seat):
    assert "full_carbon" not in obs
    y,lg,_=pol.act(x,feat.aux,teacher=2)
    alt,reason=pack._final_choice(pol,feat,x,y,lg,obs,cfg)
    baseline_actions=feat.to_actions(y)
    changed_actions=feat.to_actions(alt)
    existing=judge(full,cfg,joint)
    if existing!=rewards:
        raise ValueError(("archived official rewards mismatch",existing,rewards))
    ctrl=list(joint);ctrl[seat]=baseline_actions
    base=judge(full,cfg,ctrl)
    other=list(ctrl);other[seat]=changed_actions
    new=judge(full,cfg,other) if changed_actions!=baseline_actions else base
    gain=(new[seat]-base[seat])-(new[1-seat]-base[1-seat])
    row=dict(seat=seat,step=full["step"],reason=reason,V5_matches_logged_move=baseline_actions==joint[seat],
             baseline_wdl=int(np.sign(base[seat]-base[1-seat])),new_wdl=int(np.sign(new[seat]-new[1-seat])),
             delta_margin=round(gain,2),scores_bare=base,scores_changed=new)
    if reason=="avoid_new_plant":
        slots=np.flatnonzero(y!=alt)
        assert len(slots)==1
        slot=slots[0];uid=feat.own_uid[slot]
        row.update(worker=uid,position=obs["players"][seat][2][uid][0],action=changed_actions[uid])
    return row


def expert_candidates(pack,pol,cache):
    idx=json.loads(read_repo(ARCHIVE/"sources.json"))
    selected={g["id"] for g in json.loads(read_repo(IDENTITIES))["games"] if g["split"]=="val"}
    out=[]
    for entry in idx["games"]:
        if entry["id"] not in selected and entry["cohort"]!="later7":continue
        blob=(cache/entry["cohort"]/(entry["id"]+".json.gz")).read_bytes()
        assert sha256(blob).hexdigest()==entry["packed_sha256"]
        rep=json.loads(gzip.decompress(blob)); frames=rep["steps"];cfg=rep["configuration"]
        name=entry["id"]; kind="top5_later7" if entry["cohort"]=="later7" else "top5_val21"
        if len(frames)!=cfg["episodeSteps"]:
            out.append(dict(id=name,source=kind,early_end=len(frames)-1));continue
        joint=[p["action"] or {} for p in frames[-1]]
        full=frames[-2][0]["observation"]
        for seat in (0,1):
            feat=F.Feat(seat)
            for frame in frames[:-1]:
                obs=dict(frame[0]["observation"]);obs.update(frame[seat]["observation"])
                obs.pop("full_carbon",None)
                x=feat.observe(obs)
            row=compare(pack,pol,feat,x,obs,full,cfg,joint,rep["rewards"],seat)
            out.append(dict(id=name,source=kind,raw_sha256=entry["packed_sha256"],**row))
    return out


def original_candidates(pack,pol):
    out=[]
    # Freeze the eight completed development runs that retained joint commands at
    # this decision. Subsequent experiments must not change this cohort.
    cohorts={"basecal":2,"it50":2,"soft_it50_deep214":4,"soft_it50_v5":4,
             "soft_it50_v5_guard":4,"tree_raid":2,"v4r":2,"win_cash":4}
    for run,expected in sorted(cohorts.items()):
        report=HERE/("tree_exam_"+run)/"results.json"
        rb=json.loads(report.read_bytes())
        if rb["status"]!="complete" or len(rb["games"])!=expected:
            raise ValueError("development run incomplete: "+run)
        for game in rb["games"]:
            if not game.get("raw_file") or not game.get("valid"):raise ValueError("invalid or missing raw game")
            path=report.parent/game["raw_file"]; data=path.read_bytes()
            assert sha256(data).hexdigest()==game["raw_sha256"]
            raw=json.loads(gzip.decompress(data))
            if raw["examiner_md5"]!="57f9a594cce4f2ec9e3573d8d9d26ca1":continue
            name=f'{report.parent.name}/{raw["seed"]}/{raw["candidate_seat"]}'
            if len(raw["actions"])!=raw["game_configuration"]["episodeSteps"]-1:
                out.append(dict(id=name,source="master_development",raw_sha256=game["raw_sha256"],early_end=len(raw["actions"]))); continue
            cfg=raw["game_configuration"];seat=raw["candidate_seat"]
            sim=FastCarbon(cfg).reset(seed=raw["seed"]);feat=F.Feat(seat)
            for joint in raw["actions"][:-1]:
                feat.observe(sim.obs(seat));sim.step(*joint)
            obs=sim.obs(seat); x=feat.observe(obs)
            full=sim.full_observation()
            full["full_carbon"]=full["carbon"]
            row=compare(pack,pol,feat,x,obs,full,cfg,raw["actions"][-1],raw["official_score_by_seat"],seat)
            out.append(dict(id=name,source="master_development",raw_sha256=game["raw_sha256"],**row))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache",type=Path);parser.add_argument("out",type=Path)
    args=parser.parse_args()
    candidate=ROOT/"data/weights/sapling_1200k/soft_it50_v5_final_plant_agent.py"
    pack=module(candidate,"public_final_plant")
    w=pack._dequantize(zlib.decompress(base64.b64decode(pack._W64)),pack._QMETA)
    w["meta"]=np.array(pack._WMETA); pol=Policy(w)
    rows=expert_candidates(pack,pol,args.cache)+original_candidates(pack,pol)
    groups={}
    for group in sorted({r["source"] for r in rows}):
        all_rows=[r for r in rows if r["source"]==group]
        seen=[r for r in all_rows if "reason" in r]
        groups[group]=dict(games=len({r["id"] for r in all_rows}),ended_before_last=sum("early_end" in r for r in all_rows),
                           eligible=len(seen),reasons=dict(Counter(r["reason"] for r in seen)),changed=sum(r["reason"]=="avoid_new_plant" for r in seen),
                           wins_gained=sum(r["new_wdl"]==1 and r["baseline_wdl"]!=1 for r in seen),
                           wins_lost=sum(r["new_wdl"]!=1 and r["baseline_wdl"]==1 for r in seen),
                           margin_gains=[r["delta_margin"] for r in seen if r["reason"]=="avoid_new_plant"])
    result=dict(candidate_sha256=sha256(candidate.read_bytes()).hexdigest(),groups=groups,rows=rows,
                scope="Exact conditional last-step official scores at logged roots against logged opponent command. Only the evaluator sees the full carbon and opponent command. Top Five games are frozen prior holdouts. Previously inspected public validation and development maps are not sealed generalization tests.")
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
