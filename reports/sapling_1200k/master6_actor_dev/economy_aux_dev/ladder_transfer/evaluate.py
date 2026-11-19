"""Evaluate the frozen shared-encoder actor and matched controls on two new public cohorts."""
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
AUX=HERE.parent
ACTOR=AUX.parent
BASE=ACTOR.parent
ROOT=BASE.parents[1]
LIB=BASE/"all_top5"
sys.path.insert(0,str(AUX))
import train as experiment
base=experiment.base
sys.path.insert(0,str(ACTOR))
import data_build

P=json.loads((HERE/"protocol.json").read_bytes())
STORAGE=Path("/workspace/sapling-income-transfer-public-v1")
HISTORY_REV="2cc7c358"


def check(ok,msg):
    if not ok:
        raise ValueError(msg)


def read(p):
    data=p.read_bytes()
    return json.loads(gzip.decompress(data) if p.suffix==".gz" else data)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(p,obj):
    raw=json.dumps(obj,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()
    p.write_bytes(gzip.compress(raw,mtime=0) if p.suffix==".gz" else raw+b"\n")


def entries(p):
    return [json.loads(line) for line in gzip.decompress(p.read_bytes()).splitlines()]


def clearance():
    old_ids=set(); seeds=set(); maps=set(); hashes={}
    def register(r):
        if r.get("id"):
            old_ids.add(r["id"])
        if r.get("seed") is not None:
            seeds.add(r["seed"])
        h=r.get("initial_terrain_u8_sha256") or r.get("terrain_sha256") or r.get("terrain")
        if h:
            maps.add(h)
    old=read(ACTOR/"symmetry_dev/ladder_overlap_336e8b8c.json")
    for src in old["sources"]:
        f=BASE/src["source"];hashes[str(f.relative_to(ROOT))]=sha(f)
        for row in entries(f):
            register(row)
    check(len(old_ids)==len(seeds)==len(maps)==1475,"prior Top Five coverage changed")
    for fname in ("data_split.json","data_loss_split.json"):
        f=ACTOR/fname; hashes[str(f.relative_to(ROOT))]=sha(f)
        for g in read(f)["games"]:
            register(g)
    p=ACTOR/"data_prior_local_seeds.json";hashes[str(p.relative_to(ROOT))]=sha(p)
    seeds.update(read(p)["seeds"])
    p=ACTOR/"freight_deployment/first_response/map_preflight.json"; hashes[str(p.relative_to(ROOT))]=sha(p)
    for row in read(p)["local_native_or_training_maps"]:
        register(row)
    for p in [ACTOR/"tree_response_training/clearance.json", ACTOR/"tree_response_training/next_intake.json",
              ACTOR/"tree_response_training/supplement_01/registration.json", ACTOR/"symmetry_dev/clearance.json",
              AUX/"native_dev/clearance.json"]:
        hashes[str(p.relative_to(ROOT))]=sha(p);d=read(p)
        groups=[d["boards"]] if "boards" in d else [b["terrain_at_queue_time"] for b in d["batches"]]
        for group in groups:
            for seed,h in group.items():
                register({"seed":int(seed),"terrain":h})
    listing=subprocess.check_output(["git","ls-tree","-r","--name-only",HISTORY_REV,"--",str(BASE.relative_to(ROOT))],cwd=ROOT,text=True)
    for path in listing.splitlines():
        if not path.endswith("/results.json"):
            continue
        blob=subprocess.check_output(["git","show",HISTORY_REV+":"+path],cwd=ROOT)
        d=json.loads(blob)
        if not isinstance(d,dict):
            continue
        rows=d.get("games")
        if not isinstance(rows,list):
            continue
        for row in rows:
            if isinstance(row,dict):
                register(row)
        hashes[path]=hashlib.sha256(blob).hexdigest()
    return old_ids,seeds,maps,hashes


def selection(training):
    ids,seeds,boards,hashes=clearance()
    split=read(ACTOR/"data_split.json")["games"]
    active=set(training["sources"]["train"]["replays"])
    names={g["opponent"] for g in split if g["id"] in active}
    check(len(active)==216,"changed registered training population")
    cards=[];omitted=[]; co_ids=set();co_seeds=set();co_terrains=set()
    for tag in P["batches"]:
        batch=LIB/tag;mfile=batch/"manifest.json"; m=read(mfile)
        hashes[str(mfile.relative_to(ROOT))]=sha(mfile)
        path=batch/"validation_maps.jsonl.gz"; hashes[str(path.relative_to(ROOT))]=sha(path)
        terrain={r["id"]:r for r in entries(path)}
        fault=batch/"unscored/cards.json"
        if fault.exists():
            for gid,val in read(fault).items():
                g=val["card"]
                if "claude-AlphaCarbon-Master6" in (g.get("name_a"),g.get("name_b")):
                    omitted.append({"id":gid,"batch":tag,"reason":"terminal error or missing numeric reward","scores":g["scores"]})
        for g in m["games"]:
            if P["actor_id"] not in g["seat_strategy_ids"]:
                continue
            check(g["seat_strategy_ids"].count(P["actor_id"])==1 and g["id"] in terrain,"ambiguous seat or board")
            h=terrain[g["id"]]["initial_terrain_u8_sha256"];s=g["seat_strategy_ids"].index(P["actor_id"])
            check(g["seed"]==terrain[g["id"]]["seed"] and g["id"] not in co_ids and g["seed"] not in co_seeds and h not in co_terrains,"new-map collision")
            co_ids.add(g["id"]);co_seeds.add(g["seed"]);co_terrains.add(h)
            reasons=[]
            if not g["clean_terminal"] or not g["frame_verified"] or any(type(v) not in (float,int) or not np.isfinite(v) for v in g["score_by_seat"]):
                reasons.append("uncertified original terminal")
            if g["id"] in ids: reasons.append("prior match id")
            if g["seed"] in seeds: reasons.append("prior map seed")
            if h in boards: reasons.append("prior actual terrain")
            if reasons:
                omitted.append({"id":g["id"],"batch":tag,"reason":reasons})
                continue
            delta=g["score_by_seat"][s]-g["score_by_seat"][1-s]; opp=g["seat_names"][1-s]
            cards.append({"id":g["id"],"gid":len(cards),"seed":g["seed"],"terrain_sha256":h,"split":"transfer",
                          "sha256":g["sha256"],"turns":g["transitions"],"seat":s,"margin":delta,
                          "batch":tag,"opponent":opp,"known_opponent":opp in names,"wdl":"W" if delta>0 else "L" if delta<0 else "D"})
    return cards,{"prior_distinct_ids":len(ids),"prior_distinct_seeds":len(seeds),"prior_distinct_terrains":len(boards),
                  "training_exact_opponent_names":sorted(names),"registered_gids":sorted(active),"source_files_sha256":hashes,
                  "excluded":omitted}


def load_models(training):
    actors={};heads={};encoders={};anchors={}
    parent_path=base.WEIGHTS/"master6_actor_v1.npz"
    choices={"parent":parent_path,"control":AUX/"control.npz","candidate":AUX/"candidate.npz"}
    for arm,path in choices.items():
        expected=experiment.P["starting_actor"] if arm=="parent" else training["weights"][arm]["actor_sha256"]
        actor,_=base.model(path,expected);actor.eval(); original=base.quantized(actor).eval()
        actors[arm]=original;anchors[arm]=expected
        if arm=="parent":continue
        hp=AUX/(arm+"_readout.npz");check(sha(hp)==training["weights"][arm]["readout_sha256"],"probe changed")
        with np.load(hp) as h:
            heads[arm]={k:torch.as_tensor(h[k].copy(),dtype=torch.float32) for k in h.files}
        anchors[arm+"_readout"]=sha(hp);encoders[arm]=actor
    for field in ("center","scale"):
        check(torch.equal(heads["control"][field],heads["candidate"][field]), "head label scale changed")
    return actors,encoders,heads,anchors


@torch.no_grad()
def predict(actors,encoders,heads,data):
    probabilities={k:[] for k in actors}; income={k:[] for k in heads}
    # Same map and same within-step earlier recorded decisions for all three actors.
    for off in range(0,len(data["y"]),32):
        b=base.batch(data,np.arange(off,min(off+32,len(data["y"]))))
        stripped={k:v for k,v in b.items() if k not in ("value","carbon")}
        for k,actor in actors.items():
            lg=actor(stripped)[0]
            if off==0:
                check(torch.equal(lg,actor(b)[0]),"auxiliary-only full board or terminal field reached the action head")
                # For every slot the current label can affect only later slots.
                for slot in range(b["y"].shape[1]):
                    fake=dict(stripped);fake["y"]=b["y"].clone()
                    fake["y"][:,slot:]=0
                    before=actor(fake)[0]
                    check(torch.equal(lg[:,slot],before[:,slot]),"same or future slot action reached recorded-prefix logits")
            probabilities[k].append(torch.softmax(lg,-1).numpy())
        for k,actor in encoders.items():
            _,ctx,_=actor.encode(stripped)
            if off==0:
                check(torch.equal(ctx,actor.encode(b)[1]),"probe used a privileged feature")
                fake=dict(stripped);fake["y"]=torch.zeros_like(stripped["y"])
                check(torch.equal(ctx,actor.encode(fake)[1]),"probe used the realized actions")
            h=heads[k]; val=torch.nn.functional.linear(ctx,h["weight"],h["bias"])
            income[k].append(val.numpy())
    return {k:np.concatenate(v) for k,v in probabilities.items()},{k:np.concatenate(v) for k,v in income.items()}


def aggregate(group,data,probabilities,income,labels,mask,center,scale):
    chosen=np.asarray(group,dtype=bool);valid=base.actionable(data)&chosen[:,None]
    feature_mask=mask&chosen[:,None]; actor={};probe={}
    for name,q in probabilities.items():
        if not valid.any():continue
        nll=experiment.chosen_nll(q,data["y"]); a=q.argmax(-1)
        actor[name]={"actions":int(valid.sum()),"nll":float(nll[valid].mean()),
                     "agreement":float((a[valid]==data["y"][valid]).mean()),
                     "changed_from_parent":int(((a!=probabilities["parent"].argmax(-1))&valid).sum())}
    for name,pi in income.items():
        gold=(labels-center)/scale; diff=np.abs(pi-gold)
        err=np.where(diff<1,diff**2/2,diff-.5)
        physical=np.expm1(np.maximum(0,pi*scale+center))
        observed=np.expm1(labels)
        probe[name]={"normalized_huber":float(err[feature_mask].mean()) if feature_mask.any() else None,"by_channel":{}}
        for i,n in enumerate(TARGETS):
            use=feature_mask[:,i]
            if not use.any():continue
            probe[name]["by_channel"][n]={"valid_roots":int(use.sum()),"normalized_huber":float(err[use,i].mean()),
                "mae_cash":float(np.abs(physical[use,i]-observed[use,i]).mean()),
                "actual_mean_cash":float(observed[use,i].mean()),"predicted_mean_cash":float(physical[use,i].mean())}
    return {"roots":int(chosen.sum()),"maps":int(len(np.unique(data["gid"][chosen]))),"actor":actor,"income":probe}


TARGETS=[f"{n}_{t}" for t in (10,30) for n in ("own_delivery","own_tree","rival_delivery","rival_tree")]


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    training=read(AUX/"result.json")
    check(sha(AUX/"train.py")==training["implementation_sha256"] and sha(AUX/"protocol.json")==training["protocol_sha256"],"trained variant identity")
    cards,coverage=selection(training)
    selection_doc={"protocol_sha256":sha(HERE/"protocol.json"),"training_result_sha256":sha(AUX/"result.json"),
                   "full_prior_coverage":coverage,"cards":cards}
    write(HERE/"selection.json",selection_doc)
    STORAGE.mkdir(parents=True,exist_ok=True)
    shard={g["id"]:data_build.prep(g,STORAGE) for g in cards}
    data,hashes=base.public("transfer",cards,STORAGE,{k:v["sha256"] for k,v in shard.items()})
    labels,mask,_=experiment.targets(cards,data)
    actors,encoders,heads,anchors=load_models(training)
    probs,income=predict(actors,encoders,heads,data)
    center=heads["control"]["center"].numpy();scale=heads["control"]["scale"].numpy()
    by_map={};groups={"all":set(g["gid"] for g in cards)}
    for g in cards:
        by_map[g["id"]]=aggregate(data["gid"]==g["gid"],data,probs,income,labels,mask,center,scale)
        for label in ("seat_"+str(g["seat"]),g["wdl"],"seen_opponent" if g["known_opponent"] else "unseen_opponent",g["batch"]):
            groups.setdefault(label,set()).add(g["gid"])
    panels={}
    for label,members in groups.items():
        rows=[by_map[c["id"]] for c in cards if c["gid"] in members]
        panel=aggregate(np.isin(data["gid"],list(members)),data,probs,income,labels,mask,center,scale)
        panel["equal_map"]={"nll":{arm:float(np.mean([r["actor"][arm]["nll"] for r in rows])) for arm in actors},
            "accuracy":{arm:float(np.mean([r["actor"][arm]["agreement"] for r in rows])) for arm in actors},
            "income":{arm:float(np.mean([r["income"][arm]["normalized_huber"] for r in rows if r["income"][arm]["normalized_huber"] is not None])) for arm in heads},
            "counts_lower_than_control":{"nll":sum(r["actor"]["candidate"]["nll"]<r["actor"]["control"]["nll"] for r in rows),
              "income":sum(r["income"]["candidate"]["normalized_huber"]<r["income"]["control"]["normalized_huber"] for r in rows)}}
        panels[label]=panel
    arrays={"gid":data["gid"],"step":data["step"],"target_log1p":labels,"valid_horizon_targets":mask,"label":data["y"],
            "legal":data["legal"],"role":data["tok_kind"][:,:data["y"].shape[1]],"training_mean":center,"training_std":scale}
    arrays.update({arm+"_action_probs":q for arm,q in probs.items()})
    arrays.update({arm+"_income_standardized":v for arm,v in income.items()})
    np.savez_compressed(HERE/"predictions.npz",**arrays)
    packet={"implementation_sha256":sha(Path(__file__)),"public_builder_sha256":sha(Path(data_build.__file__)),
            "actors_and_frozen_probes_sha256":anchors,"selection_sha256":sha(HERE/"selection.json"),
            "predictions_sha256":sha(HERE/"predictions.npz"),"shard_sha256":hashes,"per_game_build":shard,
            "all_counts":dict(Counter(c["wdl"] for c in cards)),"income_targets":TARGETS,"panels":panels,"per_game":by_map}
    write(HERE/"result.json",packet)
    print("complete",len(cards),len(data["gid"]),panels["all"]["actor"],panels["all"]["equal_map"],flush=True)


if __name__=="__main__":
    main()
