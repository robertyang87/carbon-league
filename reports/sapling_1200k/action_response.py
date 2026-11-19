"""Collect matched responsive terminal action differences on preassigned, map-disjoint replay roots."""
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import orchard_branches as b

HERE=b.HERE
OUT=HERE/"action_response"
FAMILIES=("delivery","contact","planter")
FEATURES=("parent_logit_difference","loaded_home_progress","immediate_deposit","local_harvest",
          "carried_value_exposed","reachable_enemy_carried","planter_income","purchase_liquidity")
TIMES=tuple(range(60,241,15))


def hashkey(x):
    return hashlib.sha256(x.encode()).hexdigest()


def plan():
    old=json.loads((HERE/"top5_h2h_data.json").read_bytes())["games"]
    prevpaths=[HERE/"orchard_branches/plan.json",HERE/"orchard_high_branches/plan.json",
               HERE/"latest5_inventory.json"]
    used={c["seed"] for f in prevpaths for c in (lambda v:v.get("maps",v.get("games",[])))(json.loads(f.read_bytes()))}
    pool=sorted((r for r in old if r["split"]=="train" and r["seed"] not in used),
                key=lambda r:hashkey("action-response-1-map|"+r["id"]))
    cards=[]
    for i,r in enumerate(pool[:18]):
        split="train" if i<9 else "val" if i<12 else "test"
        primary=i%3
        tasks=[dict(family=FAMILIES[primary],seat=i%2)]
        if split=="train":
            tasks.append(dict(family=FAMILIES[(primary+1)%3],seat=1-i%2))
        cards.append({**{k:r[k] for k in ("id","seed","canonical_sha256")},"split":split,"tasks":tasks})
    if len(cards)!=18 or len({r["seed"] for r in cards})!=18:raise ValueError("insufficient distinct maps")
    return {"version":1,"pack":b.SHA,"maps":cards,"features":FEATURES,"grid":TIMES,
            "scope":"9 training maps (two assigned opportunities each), 3 validation and 6 one-shot test maps (one each); all selected from old train only, excluding all 30 prior orchard experiment planned maps and 26 recent diagnostic maps. Underlying networks can have seen these old demonstrations. Splits are independent for the new paired labels and the fitted head.",
            "selection":"For each assigned family and seat, among candidates at states 60..240 every 15 transitions choose minimum hash(action-response-1-root|map|seat|time|slot|alternative). At each unit use the highest frozen parent logit among role-contrasting legal alternatives. Re-decode full prefix, require unchanged non-focal orders and no friendly shared destination involving focal unit in either arm. Absence retained without replacement.",
            "families":{"delivery":"loaded collector cargo >=30, compare a step toward the own base with a stay or non-closing step; both destinations free of immediate public losing reach",
                        "contact":"collector, one option in visible enemy next-step reach and one outside. Keep defensive or proactive orientation. Enemy contact is allowed and separately recorded.",
                        "planter":"plant/seize opportunity from public observation, prefix can afford the cost; contrast staying against moving, both free of visible immediate threats"},
            "outcomes":"One root joint action changes; both sides then run the same deterministic .75 it50/.25 Sapling mixed parent from each branch's observations. U=.5*(win=1/tie=.5/loss=0)+.5*sigmoid(margin/1000) from official score. Both continuation arms independently replayed through original interpreter with step cash and status checks. These are action labels under frozen self-play continuation, not Master3s2 win-rate results.",
            "fit":"Fixed ridge lambda 10, no intercept, target alternative U minus parent U. Scale feature j by max(train RMS,.25), weight each map equally (mean within map), duplicate neither map nor action. Validation chooses threshold among 0,.02,.05 by mean map delta U, breaking ties toward higher threshold; to activate require >=2 changed validation maps, positive mean U and nonnegative mean margin. If none passes freeze retain-parent and separately expose exploratory train-only raw signs. No retuning on test.",
            "controls":"Parent always retains, static tree rule permits planter flip only on frozen >50 encouragement or <-25 avoidance and retains other families, random uniform reported as exact expected .5*delta. No deployment promotion from this small diagnostic set.",
            "selection_seed":"action-response-1","single_worker_cpu":4}


def public_costs(m,x,prefix,slot,normal,logits,legal,F,seat,obs,cfg,family):
    if slot>=10:return []
    pos=int(x["tok_pos"][slot]); dest=F.DEST[pos]; collector=x["tok_kind"][slot]==0
    if (family=="planter")==collector:return []
    cargo=float(x["tok_cargo"][slot]); aux=prefix.a
    available=legal & (prefix.reserved[dest]==0) & (prefix.undecided[dest]==0) & ~aux["en_base"][dest]
    if not available[normal]:return []
    reach=np.isfinite(aux["en_min"][dest]) | (aux["en_pl"][dest]>0)
    losing=(aux["en_min"][dest]<=cargo) if collector else reach
    distance=aux["dob"][dest]; closing=distance<aux["dob"][pos]
    raw_pos=int(F.CANON[pos]) if seat else pos
    item=None
    if family=="delivery":
        if cargo<30 or losing[normal]:return []
        valid=available & ~losing & (closing!=closing[normal])
        direction="alternative_return" if not closing[normal] else "alternative_continue"
    elif family=="contact":
        valid=available & (reach!=reach[normal])
        direction="alternative_defend" if reach[normal] else "alternative_contact"
    else:
        item=m.orchard_features(obs,obs["carbon"],seat,raw_pos,obs["step"],cfg)
        if item is None or prefix.cash<item[1] or losing[normal]:return []
        valid=available & ~losing & ((np.arange(5)==0)!=(normal==0))
        direction="alternative_buy" if normal else "alternative_move"
    valid[normal]=False
    if not np.any(valid):return []
    alt=int(np.argmax(np.where(valid,logits,-np.inf)))
    net=price=0.
    if item:
        kind,price,features=item
        net=m.orchard_predict(m._ORCHARD_MODELS[kind],features)-price
    def vector(a):
        cell=int(dest[a]); norm_cargo=min(cargo,500.)/250.
        return np.array([float(logits[a])/4.,
             norm_cargo*(float(aux["dob"][pos])-float(aux["dob"][cell]))/2 if collector else 0.,
             norm_cargo*float(aux["own_base"][cell]) if collector else 0.,
             float(aux["est"][cell])*float(collector and a==0 and not aux["own_base"][cell]),
             -norm_cargo*float(losing[a]) if collector else -float(losing[a]),
             min(float(aux["en_max"][cell]),500.)/250.*float(aux["en_max"][cell]>cargo) if collector else 0.,
             np.clip(net/120.,-2.,2.) if item and a==0 else 0.,
             -min(price/max(1.,prefix.cash),1.) if item and a==0 else 0.],float)
    diff=np.clip(vector(alt)-vector(normal),-3.,3.)
    if not np.isfinite(diff).all():raise ValueError("nonfinite public features")
    static=bool(item and ((normal!=0 and net>50) or (normal==0 and net< -25)))
    return [dict(slot=slot,normal=normal,alt=alt,direction=direction,feature=diff.tolist(),tree_static_alt=static,
                 public_price=price,tree_predicted_net=net,raw_pos=raw_pos,cargo=cargo,parent_gap=float(logits[normal]-logits[alt]))]


def find(m,F,game,card,task):
    frames=game["steps"];cfg=game["configuration"];seat=task["seat"];family=task["family"]
    opening=frames[0][0]["observation"]
    sim=b.arena.exam.FastCarbon(cfg).reset(full_carbon=opening["full_carbon"],cash=opening["players"][0][0])
    feats=[F.Feat(i) for i in (0,1)]
    best=None; opportunities=0
    last=min(max(TIMES),len(frames)-2)
    for t in range(last+1):
        obs=[b.pub(sim.obs(i)) for i in (0,1)]
        xs=[f.observe(o) for f,o in zip(feats,obs)]
        if t in TIMES:
            x=xs[seat];f=feats[seat];encoded=m._orchard_encode(x);possible=[]
            def capture(ft,st,p,slot,normal,logits,legal,changes):
                possible.extend(public_costs(m,st,p,slot,normal,logits,legal,ft,seat,obs[seat],cfg,family))
                return normal
            m._orchard_decide=capture
            base=m._orchard_decode(x,f.aux,encoded,[])
            for v in possible:
                slot,alt=v["slot"],v["alt"]
                key=hashkey(f"action-response-1-root|{card['id']}|{seat}|{t}|{slot}|{alt}")
                # Check the two exact joint actions before admitting even the first root.
                def force(ft,st,p,index,normal,logits,legal,changes):
                    if index!=slot:return normal
                    if normal!=v["normal"] or not legal[alt]:raise ValueError("invalid forced prefix")
                    return alt
                m._orchard_decide=force
                proposed=m._orchard_decode(x,f.aux,encoded,[])
                if np.count_nonzero(base!=proposed)!=1:continue
                def friendly_clear(labels):
                    ds=m._return_destinations(F,x,labels)
                    return all(dest!=ds[slot] for other,dest in ds.items() if other!=slot)
                if not friendly_clear(base) or not friendly_clear(proposed):continue
                opportunities+=1
                if best is not None and key>=best[0]:continue
                opponent=feats[1-seat].to_actions(b.decode(m,xs[1-seat],feats[1-seat]))
                root={**v,"step":t,"unit":f.own_uid[slot],"root_state_sha256":b.arena.prior.hashed(sim.get_state())}
                best=(key,root,sim.get_state(),copy.deepcopy(feats),f.to_actions(base),f.to_actions(proposed),opponent)
        if t<last:
            sim.step(*(player["action"] or {} for player in frames[t+1]))
            if list(sim.cash)!=[p[0] for p in frames[t+1][0]["observation"]["players"]]:
                raise ValueError("training tape cash mismatch")
    return best,opportunities


def utility(margin):
    w=1. if margin>0 else 0. if margin<0 else .5
    return .5*w+.5/(1.+math.exp(-max(-60.,min(60.,margin/1000.))))


def resultfile(card,task):
    return OUT/(card["id"]+"_"+task["family"]+".json.gz")


def work(card,task,paths,plan_hash):
    target=resultfile(card,task)
    if target.exists():
        g=json.loads(gzip.decompress(target.read_bytes()))
        if (g["card"],g["task"],g["plan_sha256"])!=(card,task,plan_hash):raise ValueError("stale output")
        return
    game=json.loads(gzip.decompress(paths[card["id"]].read_bytes()))
    if hashlib.sha256(b.canonical(game)).hexdigest()!=card["canonical_sha256"]:raise ValueError("incorrect replay")
    m=b.arena.prior.shared.module(b.PACK,"action_response_hybrid")
    m.agent(b.pub(game["steps"][0][0]["observation"]),game["configuration"])
    F=sys.modules["cnet2_feat"]
    best,n=find(m,F,game,card,task)
    data={"card":card,"task":task,"pack_sha256":b.SHA,"plan_sha256":plan_hash,"eligible_opportunities":n}
    if best is not None:
        _,root,snap,feats,base,alt,opp=best;seat=task["seat"]
        data.update(root=root)
        for tag,orders in (("parent",base),("alternative",alt)):
            data[tag]=b.rollout(m,game["configuration"],game["steps"],root,snap,feats,orders,opp,seat)
        dm=data["alternative"]["margin"]-data["parent"]["margin"]
        data["delta_margin"]=dm
        data["delta_u"]=utility(data["alternative"]["margin"])-utility(data["parent"]["margin"])
    b.write(target,data,True)
    print(card["split"],card["id"][:8],task["family"],"label" if best is not None else "no-root",flush=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument("--replays",type=Path,required=True)
    cli.add_argument("--stage",choices=("trainval","test"),required=True);args=cli.parse_args()
    os.sched_setaffinity(0,{4})
    if b.arena.prior.digest(b.PACK)!=b.SHA:raise ValueError("unexpected parent")
    OUT.mkdir(exist_ok=True);expected=json.loads(json.dumps(plan()));fp=OUT/"plan.json"
    if fp.exists() and json.loads(fp.read_bytes())!=expected:raise ValueError("plan changed")
    if not fp.exists():b.write(fp,expected)
    plan_hash=hashlib.sha256(fp.read_bytes()).hexdigest()
    if args.stage=="test":
        freeze=json.loads((OUT/"freeze.json").read_bytes())
        if b.arena.prior.digest(OUT/"model.json")!=freeze["model_sha256"] or freeze["plan_sha256"]!=plan_hash:
            raise ValueError("model must be frozen before opening test maps")
    paths={f.name.removesuffix(".json.gz"):f for f in args.replays.rglob("*.json.gz")}
    for card in expected["maps"]:
        if (args.stage=="test")!=(card["split"]=="test"):continue
        for task in card["tasks"]:work(card,task,paths,plan_hash)


if __name__=="__main__":main()
