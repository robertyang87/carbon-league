"""Last-step causal check on pre-existing legal games; enemy simultaneous order fixed by tape."""
import copy
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parents[1]
sys.path.insert(0,str(ACTOR.parents[2]/"code"))
from fastsim import FastCarbon
from eval_v53 import _extinct


def read_agent(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def run():
    parent=ACTOR/"actor_partner_dev/candidate.py"; candidate=HERE/"candidate.py"
    sha=lambda data:hashlib.sha256(data).hexdigest()
    actors=[read_agent("role_last_old",parent),read_agent("role_last_new",candidate)]
    rows=[]
    for f in sorted((ACTOR/"first_win_dev/games/candidate").glob("raw_*.json.gz")):
        raw=f.read_bytes();g=json.loads(gzip.decompress(raw));seat=g["candidate_seat"]
        if g["pack_sha256"]!=sha(parent.read_bytes()) or g["steps"]!=299 or g["errors"]:
            raise ValueError(f"unexpected logged game {f.name}")
        sim=FastCarbon(g["game_configuration"]).reset(seed=g["seed"])
        if list(sim.carbon)!=g["initial_full_carbon"]:raise ValueError("map mismatch")
        for t,joint in enumerate(g["actions"][:-1]):
            obs=sim.obs(seat)
            for i,m in enumerate(actors):
                if t==0:
                    if m.agent(obs,sim.config)!=joint[seat]:raise ValueError("opening mismatch")
                else:m._FEAT.observe(obs)
            sim.step(*joint)
        obs=sim.obs(seat)
        actual=[m.agent(obs,sim.config) for m in actors]
        logged=g["actions"][-1]
        if actual[0]!=logged[seat]:raise ValueError("parent's terminal decision does not match raw")
        cash_before=list(sim.cash)
        results=[]
        for act in actual:
            branch=copy.deepcopy(sim)
            moves=logged.copy();moves[seat]=act
            branch.step(*moves)
            post=branch.get_state()
            dead=[_extinct(post,p) for p in (0,1)]
            # In this corpus all branches must be normal step-299 endings.
            if any(dead) or branch.step_count!=299: raise ValueError("non-normal terminal branch")
            score=list(branch.cash)
            results.append(float(score[seat]-score[1-seat]))
        if abs(results[0]-g["margin"])>1e-5:raise ValueError((g["margin"],results[0],"terminal margin mismatch"))
        rows.append(dict(seed=g["seed"],seat=seat,t=298,source_sha256=sha(raw),cash_before=cash_before,
                         commands=actual,external_command=logged[1-seat],margins=results,
                         delta=round(results[1]-results[0],5)))
    if len(rows)!=16:raise ValueError("incomplete archive")
    report=dict(parent_sha256=sha(parent.read_bytes()),candidate_sha256=sha(candidate.read_bytes()),
                maps=8,roots=len(rows),changed=sum(r["commands"][0]!=r["commands"][1] for r in rows),
                mean_margin_delta=float(np.mean([r["delta"] for r in rows])),
                won_by_change=sum(r["margins"][1]>0>=r["margins"][0] for r in rows),
                scope="exact last move only with pre-recorded simultaneous real opponent command, no full candidate games",rows=rows)
    with (HERE/"terminal.json").open("x") as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(report["changed"],report["mean_margin_delta"],flush=True)


if __name__=="__main__":run()
