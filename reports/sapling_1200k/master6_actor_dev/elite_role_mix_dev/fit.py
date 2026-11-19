"""Fit deployment logit shares on game-level expert action agreement with fixed phases."""
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parent
TRAIN = ACTOR / "blended_actor_dev"
spec = importlib.util.spec_from_file_location("blend_reference", TRAIN / "fit.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
sys.path.insert(0, str(ACTOR))
import data_build

PARENT = ACTOR / "actor_partner_dev/candidate.py"
WEIGHTS = [0.05, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85]
PHASES = [90, 210]
MASTER6 = "claude-AlphaCarbon-Master6"
TOP5 = ACTOR.parent / "all_top5"
CACHE = Path("/workspace/sapling-all-top5-20261119/replays")
NEW_SHARDS = Path("/tmp/sapling-role-mix-verified-shards-v1")
ROLE = ("collector", "planter", "base")
NEW_MAPS_PER_SEAT = 12


def save(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def new_maps(old):
    seen = {k: {g[k] for g in old} for k in ("seed", "terrain_sha256")}
    union = {}
    for manifest in sorted(TOP5.glob("appendix_*/manifest.json")):
        for g in json.loads(manifest.read_bytes())["games"]:
            if (g["seat_names"].count(MASTER6) == 1 and g["clean_terminal"] and not g["self_play"]
                    and g["seed"] not in seen["seed"] and (CACHE / (g["id"]+".json.gz")).is_file()):
                union[g["id"]] = g
    key = lambda g: hashlib.sha256(("role-mix-map-v1|" + g["id"]).encode()).hexdigest()
    selected = []
    for seat in (0, 1):
        group = []
        for g in sorted(union.values(), key=key):
            if g["seat_names"][seat] != MASTER6 or g["seed"] in seen["seed"]:
                continue
            packed = (CACHE / (g["id"]+".json.gz")).read_bytes()
            if hashlib.sha256(packed).hexdigest() != g["sha256"]:
                raise ValueError("replay bytes changed")
            env = json.loads(gzip.decompress(packed))
            land = hashlib.sha256(bytes(env["steps"][0][0]["observation"]["full_carbon"])).hexdigest()
            if land in seen["terrain_sha256"]:
                continue
            seen["terrain_sha256"].add(land)
            seen["seed"].add(g["seed"])
            margin = g["score_by_seat"][seat] - g["score_by_seat"][1-seat]
            group.append(dict(id=g["id"], sha256=g["sha256"], terrain_sha256=land, seed=g["seed"],
                              seat=seat, turns=g["transitions"], margin=margin,
                              opponent=g["seat_names"][1-seat], split="external", gid=len(selected)+len(group),
                              wdl="W" if margin > 0 else "D" if margin == 0 else "L"))
            if len(group) == NEW_MAPS_PER_SEAT:
                break
        if len(group) != NEW_MAPS_PER_SEAT:
            raise ValueError("not enough independent maps")
        selected.extend(group)
    return selected


@torch.inference_mode()
def measure(data, models, indices):
    fields = {k:[] for k in ("gid","step","role","y","expert","student")}
    for start in range(0, len(indices), 24):
        ix = indices[start:start+24]
        b = ref.common.batch(data, ix)
        mask = ref.common.actionable(b).numpy()
        y = b["y"].numpy()
        for k, values in (("gid",data["gid"][ix,None]), ("step",data["step"][ix,None]),
                          ("role",data["tok_kind"][ix,:y.shape[1]]), ("y",y)):
            fields[k].append(np.broadcast_to(values, y.shape)[mask])
        for k, net in zip(("expert","student"),models):
            fields[k].append(net(b)[0].numpy()[mask])
    return {k:np.concatenate(v) for k,v in fields.items()}


def metrics(f, alpha, idx):
    y = f["y"][idx]; g=f["gid"][idx]
    if not len(g):
        raise ValueError("empty group")
    mix = np.broadcast_to(alpha, f["gid"].shape)[idx,None]
    scores = (1.-mix)*f["expert"][idx]+mix*f["student"][idx]
    correct = (scores.argmax(-1)==y)
    gid, rev = np.unique(g,return_inverse=True)
    rates=np.bincount(rev,weights=correct)/np.bincount(rev)
    return {"maps":len(gid),"slots":len(y),"accuracy":float(rates.mean())}, rates


def apply(f, weights):
    phase=np.searchsorted(PHASES,f["step"],side="right")
    w=np.asarray(weights)[phase,f["role"]]
    return np.where(f["step"]==0,.25,w)


def summary(f, weights):
    alpha=apply(f,weights)
    out={}
    for k, idx in {"all":np.ones(len(alpha),bool), **{s:f["role"]==r for r,s in enumerate(ROLE)}}.items():
        base, br=metrics(f,.25,idx); mix,mr=metrics(f,alpha,idx)
        diff=mr-br
        draws=np.random.default_rng(348115).choice(diff,(5000,len(diff)),replace=True).mean(1)
        out[k]={"active":base,"candidate":mix,"paired_map_delta":float(diff.mean()),
                "map_bootstrap_95":np.quantile(draws,[.025,.975]).tolist()}
    return out


def package(weights):
    old=PARENT.read_text()
    anchor='combined = _HYBRID["expert_weight"] * scores[0] + _HYBRID["student_weight"] * scores[1]'
    if old.count(anchor)!=2: raise ValueError("unexpected decoder")
    new=old.replace(anchor, 'alpha = _role_mix_share(x, slot)\n        combined = (1.0-alpha) * scores[0] + alpha * scores[1]')
    new+="\n_ROLE_MIX_WEIGHTS = "+repr(weights)+"\n"
    new+="""def _role_mix_share(x, slot):
    step = 299 - x["left"]
    if step == 0:
        return 0.25
    phase = int(step >= 90) + int(step >= 210)
    return _ROLE_MIX_WEIGHTS[phase][int(x["tok_kind"][slot])]
"""
    if len(new.encode())>=10_000_000:raise ValueError("package size limit")
    path=HERE/"candidate.py";path.write_text(new);return path


def main():
    torch.set_num_threads(1)
    games=json.loads((TRAIN/"panel.json").read_bytes())["games"]
    tr=[g for g in games if g["split"]=="train"]; va=[g for g in games if g["split"]=="val"]; ho=[g for g in games if g["split"]=="holdout"]
    if list(map(len,(tr,va,ho)))!=[218,10,18] or ref.sha(PARENT)!=ref.PROTOCOL["partner_pack_sha256"]:
        raise ValueError("map split or parent changed")
    fresh=new_maps(games)
    save(HERE/"protocol.json",dict(parent=ref.sha(PARENT),train_ids=[g["id"] for g in tr],external_games=fresh,
         train_roots="16 roots/map randomly across existing verified roots, seed 550211",phase_boundaries=PHASES,shares=WEIGHTS,
         select="train-only average within-map agreement; retain .25 unless best >= +0.5pp and positive lower 90% bound; smallest change within .15pp of best; keep t0=.25",
         gate="external 24 maps: teacher-forced and live-prefix map-mean agreement above parent; live changes >= 5% roots; no efficacy inference",
         external_algorithm="hash order per physical seat, remove all 246 actor maps by seed and terrain, 12 maps each seat"))
    models=[ref.expert_model(),ref.common.quantized(ref.common.model(ref.PARENT,ref.PROTOCOL["parent_weights_sha256"])[0]).eval()]
    data=ref.load(tr); rng=np.random.default_rng(550211)
    ix=np.concatenate([rng.choice(np.flatnonzero(data["gid"]==g["gid"]),min(16,g["rows"]),replace=False) for g in tr])
    f=measure(data,models,ix); phase=np.searchsorted(PHASES,f["step"],side="right")
    weights=[]; grid={}
    for ph in range(3):
        row=[]
        for role,name in enumerate(ROLE):
            m=(f["role"]==role)&(phase==ph)&(f["step"]>0)
            curves=[metrics(f,w,m) for w in WEIGHTS]
            base=curves[2][1]; ranked=np.array([x[0]["accuracy"] for x in curves]); best=ranked.max()
            allowed=[j for j in range(len(curves)) if ranked[j]>=best-.0015 and
                     (curves[j][1]-base).mean()>=.005 and np.quantile(
                     rng.choice(curves[j][1]-base,(1000,len(base)),replace=True).mean(1),.05)>0]
            choice=min(allowed,key=lambda j:abs(WEIGHTS[j]-.25)) if allowed else 2
            row.append(WEIGHTS[choice]);grid[f"{ph}/{name}"]={"results":[c[0] for c in curves],"choice":row[-1]}
        weights.append(row)
    candidate=package(weights)
    result={"fit_grid":grid,"shares":weights,"source_sha256":ref.sha(PARENT),"candidate_sha256":ref.sha(candidate),
            "candidate_bytes":candidate.stat().st_size,"train":summary(f,weights)}
    for k,rows in (("validation",va),("historical_holdout",ho),("external",fresh)):
        if k=="external":
            NEW_SHARDS.mkdir(exist_ok=True)
            rows=[dict(g,shard_sha256=r["sha256"],rows=r["rows"]) for g in rows for r in [data_build.prep(g,NEW_SHARDS)]]
        data=ref.load(rows,NEW_SHARDS) if k=="external" else ref.load(rows)
        f=measure(data,models,np.arange(len(data["gid"])))
        result[k]=summary(f,weights)
    save(HERE/"fit.json",result)
    print(weights,result["external"]["all"],flush=True)


if __name__=="__main__":
    main()
