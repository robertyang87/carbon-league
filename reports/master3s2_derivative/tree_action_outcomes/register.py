"""Freeze four old-map intervention roots, using only active-seat public information and parent issued packets."""
import hashlib,io,json,gzip,zipfile,sys
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
HERE=Path(__file__).resolve().parent;D=HERE.parent
sys.path.insert(0,str(D/"tree_control_response/isolated_proposals"))
from order import Anchor
def load(name,path):
    s=spec_from_file_location(name,path);m=module_from_spec(s);s.loader.exec_module(m);return m
lib=load("plant_features",D/"ladder/tree_payback/runtime.py")
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())

def used_roots():
    found=set()
    def walk(v):
        if isinstance(v,list):
            for z in v:walk(z)
        elif isinstance(v,dict):
            z=[v.get("seed",v.get("map")),v.get("seat"),v.get("step",v.get("t",v.get("root_step",v.get("starting_step"))))]
            if all(type(k) is int for k in z) and 621758592<=z[0]<=621758595:found.add(tuple(z))
            for o in v.values():walk(o)
    files=set()
    for glob in ("*protocol.json","roots.json","selection.json","*_result.json"):
        files.update(D.rglob(glob))
    for p in files:
        if HERE in p.parents:continue
        walk(read(p))
    return found

def main():
    w=read(D/"tree_control_response/weights.json");risk=read(D/"ladder/tree_payback/weights.json")["full"]
    cfg=read(D.parents[1]/"code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json")["configuration"]
    cfg={k:v["default"] for k,v in cfg.items() if isinstance(v,dict) and "default" in v}
    cfg.setdefault("episodeSteps",300)
    banned=used_roots(); maps={}
    for lane in ("prediction_feedback","cargo_defense"):
        for file in sorted((D/f"ladder/{lane}/development/hosted").glob("*.zip")):
            with zipfile.ZipFile(file) as z:
                if "parent/registration.json" not in z.namelist():continue
                reg=json.loads(z.read("parent/registration.json"));rc=json.loads(z.read("parent/receipt.json"))
                s=reg["candidate_seat"];seed=reg["seed"];raw=z.read("parent/observations.jsonl.gz");tr=z.read("parent/turns.jsonl")
                assert reg["files"]["candidate"]["sha256"].startswith("7cf5c6c9") and reg["files"]["opponent"]["sha256"].startswith("65f268b2")
                assert rc["turns_sha256"]==sha(tr) and len(gzip.decompress(raw).splitlines())==len(tr.splitlines())==299
                frames=[json.loads(l) for l in gzip.decompress(raw).splitlines()]
                turns=[json.loads(l) for l in tr.splitlines()]
                sig=(seed-621758592)%2==1
                for t in range(272,287):
                    if t in (279,285) or (seed,s,t) in banned:continue
                    row=turns[t];ob=frames[t]["observations"][s];own,enemy=ob["players"][s],ob["players"][1-s]
                    hist={k:frames[t-k]["observations"][s]["players"] for k in (3,5)}
                    blocked={c for p in ob["players"] for c in p[1].values()}|{a[0] for p in ob["players"] for a in p[3].values()}
                    issued=row["calls"][s]["response"]["action"]
                    for uid,(cell,cargo,role) in sorted(own[2].items()):
                        if role!="PLANTER" or cell in blocked:continue
                        a=Anchor(ob,issued,uid,cfg); choices=a.candidates();original=issued.get(uid,"NONE")
                        if a.cash_after_recruits<3*lib.quoted_price(ob["players"],s,cfg) or a.other_destinations.get(cell):continue
                        if original!="NONE" and not choices["NONE"]["accepted"]:continue
                        x=lib.features(ob,uid,cfg,hist)
                        z=[(v-m)/vscale for v,m,vscale in zip(x,w["feature_mean"],w["feature_scale"])]
                        score,hold=[mu+sum(a*b[j] for a,b in zip(z,w["coef"])) for j,mu in enumerate(w["means"])]
                        if (score>0)!=sig or abs(score)<1:continue
                        if original=="NONE":
                            space=[(min((lib.distance(r["destination"],j[0],cfg["size"]) for j in enemy[2].values()),default=15),-i,k) for i,(k,r) in enumerate(choices.items()) if r["accepted"] and r["destination"] not in blocked]
                            if not space:continue
                            leave=max(space)[-1]
                        else:leave=original
                        stay=issued.copy();stay.pop(uid,None);depart={**issued,uid:leave}
                        check=choices["NONE" if original!="NONE" else leave]
                        assert check["candidate_order"]==(stay if original!="NONE" else depart) and check["accepted"]
                        entry=dict(seed=seed,seat=s,step=t,uid=uid,cell=cell,kind="vacant_new_plant",historical=original,leave_direction=leave,archive=file.stem,
                            archive_relpath=str(file.relative_to(D)),archive_sha256=sha(file.read_bytes()),public_stream_sha256=sha(raw),public_root_hashes=row["public_observation_sha256"],
                            terrain=reg["physical_map_sha256"],stay=stay,leave=depart,treasury=own[0],rng=[(seed+83731+17*i)%2**32 for i in (0,1)],
                            execution_order=["stay","leave"] if seed%2==0 else ["leave","stay"],estimated_net=score,estimated_hold=hold,estimated_nonpayback=lib.predict(x,risk),
                            caller_observation_sha=ob and a.receipt()["public_sha256"],quote=lib.quoted_price(ob["players"],s,cfg),focal_delta=check["changed"])
                        maps.setdefault(seed,[]).append(entry)
    tickets=[]
    for seed in range(621758592,621758596):
        # Mix seats and choose an early root with time left. Do not inspect responses or future outcomes for selection.
        opts=maps.get(seed,[])
        chosen=min(opts,key=lambda v:(v["seat"]!=(seed%2),v["step"],v["uid"])) if opts else None
        if chosen:tickets.append(chosen)
        print(seed,len(opts),None if not chosen else [chosen[k] for k in ("seat","step","uid","estimated_net","historical")])
    doc=dict(schema="original-two-arm-late-planter/1",candidate="7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed",original="65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5",
             selection="Four pre-existing physical maps 592/593/594/595. Even map forecast negative old trained initial-tree net, odd positive, |net|>=1. Archived parent seat preferred by seed parity; then earliest t272..286 except 279/285 and previous registered branch roots, ascending UID. Off-base empty candidate, stationary and alternative legal under complete parent order (3x quote post-recruit cash). Leave = old physical command when moving or else safest empty cell by nearest enemy worker distance. No future reward or actual enemy current action used.",
             skipped_existing_tickets=sorted(banned),predictors_sha=sha((D/"tree_control_response/weights.json").read_bytes()),fit_games=24,held_games=8,
             agreement_check="After label collection test sign of estimated first-tenure tree net vs genuine official stay minus leave. A one-dimensional centered ridge calibration with intercept and lambda=1 per training map will be trained three/map-held-out for each eligible complete new root; compare cross-validation to train-only constant and to unadjusted old estimate. Four selected maps, one paired follow-up per map; descriptive and train-only.",
             registration_source=sha(Path(__file__).read_bytes()),tickets=tickets,absent_seeds=[v for v in range(621758592,621758596) if v not in maps],gate="No added maps or reselection; no formal match. The sole divergence at root is the named planter, same original enemy action. Shared post-root cloned policy/official state, with private streams and measured search cost explicitly reset to identical values per arm, followed by genuine public responses.")
    assert len({t["terrain"] for t in tickets})==len(tickets) and not {t["terrain"] for t in tickets}&(set(w["train_maps"])|set(w["held_maps"]))
    out=(json.dumps(doc,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode();p=HERE/"protocol.json";assert not p.exists() or p.read_bytes()==out;p.write_bytes(out);print(sha(out))
if __name__=="__main__":main()
