"""Fixed-map paired, state-responsive terminal comparison of planting now versus the hybrid's move."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import tree_audited_game as arena
from official_game_replay import make
from zerosum_env.utils import structify
from orchard_train import canonical
from turning_point_ledger import CashEvents

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACK = ROOT/"data/weights/sapling_1200k/orchard_it50_v3.py"
SHA = "1dc7345ea71ab692e3db09c00b0958f5fc086302493c5301afa89eb97125b38d"
OUT = HERE/"orchard_branches"


def write(path, obj, compress=False):
    data=json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=None if compress else 2).encode()
    temp=path.with_suffix(".tmp")
    temp.write_bytes(gzip.compress(data,mtime=0) if compress else data)
    os.replace(temp,path)


def plan():
    cards = json.loads((HERE/"top5_h2h_data.json").read_bytes())["games"]
    new = json.loads((HERE/"latest5_inventory.json").read_bytes())["games"]
    if {c["seed"] for c in cards}&{c["seed"] for c in new}:
        raise ValueError("new diagnostic map overlaps original demonstration split")
    pool=sorted((c for c in cards if c["split"]=="train"),
                key=lambda c: hashlib.sha256(("orchard-paired-1|"+c["id"]).encode()).digest())
    rows=[]
    for i,card in enumerate(pool[:6]):
        rows.append({k:card[k] for k in ("id","seed","canonical_sha256")})
        rows[-1].update(seat=i%2, group="design" if i<3 else "reserved", center=[75,105,135][i%3])
    return {"pack":SHA,"maps":rows,"map_selection":"First six old training maps by SHA256(orchard-paired-1|id), exactly one prospective root per map. First three design, last three reserved. No training or deployment changes. All 26 recent maps excluded.",
            "root_selection":"At steps 60..150 every fifth step, among feasible clean new plant vs original neural move, keep highest public predicted net value, resolving ties by closest to fixed center then earliest step, smallest slot. Both arms require safe destinations, actual prefix cash, no new friendly collision. No terminal outcomes enter selection.",
            "arms":"Both seats continue as the same deterministic unmodified .75/.25 learned-mixture parent responding to new observations. The moving root is the original parent. The buying root changes a single planter to stay and re-decodes the remaining real prefix; both share the same opposing root command.",
            "decision":"Descriptive pilot. No promotion; a positive reserved average alone cannot establish generalization or official Master3s2 strength."}


def pub(obs):
    return {k:v for k,v in obs.items() if k!="full_carbon"}


def decode(m, x, f):
    return m._orchard_decode(x, f.aux, m._orchard_encode(x))


def choose(m, F, frames, cfg, seat, center, high_threshold=None):
    init=frames[0][0]["observation"]
    sim=arena.exam.FastCarbon(cfg).reset(full_carbon=init["full_carbon"],cash=init["players"][0][0])
    feats=[F.Feat(i) for i in (0,1)]
    best=None
    inspected=0
    last_step=min(150,len(frames)-2)
    for t in range(last_step+1):
        xs=[f.observe(pub(sim.obs(i))) for i,f in enumerate(feats)]
        window=60<=t<=150 and t%5==0
        if window and high_threshold is not None:
            obs=pub(sim.obs(seat))
            public_candidates=[m.orchard_features(obs,obs["carbon"],seat,cell,t,cfg)
                               for cell,_,role in obs["players"][seat][2].values() if role=="PLANTER"]
            window=any(info and info[0]=="plant" and
                       m.orchard_predict(m._ORCHARD_MODELS["plant"],info[2])-info[1]>high_threshold
                       for info in public_candidates)
        if window:
            x,f=xs[seat],feats[seat]
            encoded=m._orchard_encode(x)
            possible=[]
            def collect(ft, st, p, slot, normal, logits, legal, changes):
                if slot>=10 or st["tok_kind"][slot]!=1 or not normal:
                    return normal
                pos=int(st["tok_pos"][slot])
                cell=int(ft.CANON[pos]) if seat else pos
                obs=pub(sim.obs(seat))
                offer=m.orchard_features(obs,obs["carbon"],seat,cell,t,cfg)
                if offer is None or offer[0]!="plant" or p.cash<offer[1]:
                    return normal
                safe=legal & (p.reserved[ft.DEST[pos]]==0) & (p.undecided[ft.DEST[pos]]==0)
                safe &= ~np.isfinite(p.a["en_min"][ft.DEST[pos]]) & (p.a["en_pl"][ft.DEST[pos]]==0)
                if safe[0] and safe[normal]:
                    value=m.orchard_predict(m._ORCHARD_MODELS["plant"],offer[2])-offer[1]
                    if high_threshold is not None and value<=high_threshold:
                        return normal
                    possible.append((slot,normal,float(value),float(offer[1]),cell))
                return normal
            m._orchard_decide=collect
            base=m._orchard_decode(x,f.aux,encoded,[])
            for slot,normal,value,price,cell in possible:
                def buy(ft, st, p, index, standard, logits, legal, changes):
                    if index==slot:
                        if standard!=normal or not legal[0]:
                            raise ValueError("prefix changed before focal slot")
                        changes.append((index,standard,0))
                        return 0
                    return standard
                changes=[]
                m._orchard_decide=buy
                proposed=m._orchard_decode(x,f.aux,encoded,changes)
                if not m._return_joint_ok(F,x,f.aux,base,proposed,changes):
                    continue
                if high_threshold is not None and np.count_nonzero(base!=proposed)!=1:
                    continue
                inspected+=1
                key=(value,-abs(t-center),-t,-slot)
                if best is not None and best[0]>=key:
                    continue
                opponent=feats[1-seat].to_actions(decode(m,xs[1-seat],feats[1-seat]))
                best=(key,dict(step=t,unit=f.own_uid[slot],cell=cell,public_price=price,
                               predicted_net=value,slot=slot,root_state_sha256=arena.prior.hashed(sim.get_state())),
                      sim.get_state(),copy.deepcopy(feats),f.to_actions(base),f.to_actions(proposed),opponent)
        if t<last_step:
            sim.step(*(p["action"] or {} for p in frames[t+1]))
            if list(sim.cash)!=[p[0] for p in frames[t+1][0]["observation"]["players"]]:
                raise ValueError("demonstration states could not be reconstructed")
    return best,inspected


def official(frames, cfg, root, tail, score, expected_cash):
    # Starting from the actual public match's opening keeps its board exact.
    # Creating an ordinary environment first also resets the upstream ID counters.
    env=make("carbon",configuration=cfg)
    env.states=structify(copy.deepcopy(frames[0]))
    env.steps=[env.states]
    seq=[[f["action"] or {} for f in turn] for turn in frames[1:root+1]]+tail
    for i,actions in enumerate(seq):
        if env.done:
            raise ValueError("recorded branch continues after official termination")
        env.step(actions)
        if i+1<=root and env.states[0].observation.players!=frames[i+1][0]["observation"]["players"]:
            raise ValueError(f"official historical players differ at {i+1}")
        if i+1>=root and i+1-root<len(expected_cash):
            observed=[p[0] for p in env.states[0].observation.players]
            if observed!=expected_cash[i+1-root]:raise ValueError(f"official step cash mismatch {i+1}")
    if not env.done or [p.reward for p in env.states]!=score:
        raise ValueError(f"official terminal mismatch: {score}, {[p.reward for p in env.states]}")
    return [p.status for p in env.states]


def rollout(m, cfg, frames, root, snapshot, feats, first, opp, seat):
    sim=arena.exam.FastCarbon(cfg)
    sim.set_state(snapshot)
    sim.cash=CashEvents(sim.cash)
    feats=copy.deepcopy(feats)
    log=[]
    cash=[list(sim.cash)]
    totals=[Counter(),Counter()]
    owned=None
    tenure=None
    while sim.step_count<cfg["episodeSteps"]-1:
        pre=list(sim.cash)
        if not log:
            actions=[None,None]
            actions[seat],actions[1-seat]=first,opp
        else:
            actions=[]
            for i,f in enumerate(feats):
                x=f.observe(pub(sim.obs(i)))
                actions.append(f.to_actions(decode(m,x,f)))
        sim.cash.events=[]
        sim.step(*actions)
        log.append(actions)
        cash.append(list(sim.cash))
        for kind,owner,val,*info in sim.cash.events:
            totals[owner][kind]+=val
            if len(log)==1 and owner==seat and kind=="plant" and info[0]==root["unit"]:
                owned,=[tid for tid in sim.player_trees[seat] if sim.trees[tid].pos==info[1]]
                tenure=dict(id=owned,cost=-val,acquired_state=sim.step_count,last_owned_state=sim.step_count,
                            absorption=0.,collision_cash=0.,termination="terminal")
            if owned and kind=="tree_absorption" and info[0]==owned and owner==seat:
                tenure["absorption"]+=val
            if owned and kind=="tree_collision_cash" and info[2]==owned and owner==seat:
                tenure["collision_cash"]+=val
        if owned and owned not in sim.player_trees[seat]:
            tenure["termination"]="seizure" if owned in sim.trees else "expiry"
            owned=None
        if owned:
            tenure["last_owned_state"]=sim.step_count
        post=sim.get_state()
        extinct=[bool(arena.exam._extinct(post,i)) for i in (0,1)]
        if any(extinct):
            break
    n=sim.step_count
    score=[n-cfg["episodeSteps"]-1 if extinct[i] else (pre if any(extinct) else sim.cash)[i] for i in (0,1)]
    status=official(frames,cfg,root["step"],log,score,cash)
    if tenure:tenure["net"]=tenure["absorption"]+tenure["collision_cash"]-tenure["cost"]
    return dict(score_by_seat=score,margin=score[seat]-score[1-seat],terminal_state=n,extinct=extinct,
                terminal_status=status,root_tree=tenure,by_seat=totals,actions=log,
                terminal_state_sha256=arena.prior.hashed(post))


def work(card, paths, cpu, out_dir=OUT, high_threshold=None):
    os.sched_setaffinity(0,{cpu})
    dest=out_dir/(card["id"]+".json.gz")
    if dest.exists():
        old=json.loads(gzip.decompress(dest.read_bytes()))
        if old["card"]!=card or old["pack_sha256"]!=SHA:raise ValueError("existing result identity differs")
        return "root" in old
    game=json.loads(gzip.decompress(paths[card["id"]].read_bytes()))
    if hashlib.sha256(canonical(game)).hexdigest()!=card["canonical_sha256"]:raise ValueError("source hash mismatch")
    cfg=game["configuration"]
    m=arena.prior.shared.module(PACK,"orchard_paired_hybrid")
    frames=game["steps"]
    m.agent(pub(frames[0][0]["observation"]),cfg)
    F=sys.modules["cnet2_feat"]
    out={"card":card,"pack_sha256":SHA}
    best,out["eligible_slots"]=choose(m,F,frames,cfg,card["seat"],card["center"],high_threshold)
    if best:
        _,root,sim,feats,move,buy,opp=best
        out["root"]=root
        for label,first in (("move",move),("buy",buy)):
            out[label]=rollout(m,cfg,frames,root,sim,feats,first,opp,card["seat"])
        out["buy_minus_move"]=out["buy"]["margin"]-out["move"]["margin"]
    write(dest,out,True)
    print(card["group"],card["seat"],card["id"][:8],out.get("buy_minus_move"),flush=True)
    return bool(best)


def run_lane(cards,paths,cpu):
    for card in cards:work(card,paths,cpu)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--replays",type=Path,required=True)
    ap.add_argument("--cpus",nargs=2,type=int,default=[0,1]);args=ap.parse_args()
    if arena.prior.digest(PACK)!=SHA:raise ValueError("pack identity mismatch")
    p=plan();OUT.mkdir(exist_ok=True)
    file=OUT/"plan.json"
    if file.exists() and json.loads(file.read_bytes())!=p:raise ValueError("refusing to replace original design")
    if not file.exists():write(file,p)
    paths={x.name.removesuffix(".json.gz"):x for x in args.replays.rglob("*.json.gz")}
    if not set(c["id"] for c in p["maps"])<=set(paths):raise ValueError("missing frozen training replays")
    with ProcessPoolExecutor(2) as pool:
        futures=[pool.submit(run_lane,p["maps"][idx::2],paths,cpu) for idx,cpu in enumerate(args.cpus)]
        for f in futures:f.result()
