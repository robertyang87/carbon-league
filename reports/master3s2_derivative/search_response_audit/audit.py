"""Measure actual draw-index drift between captured sibling rollouts, without new games."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent
sys.path.insert(0,str(FAMILY/"opponent_policy"))
from original_policy import module,original

def sha(b):
    return hashlib.sha256(b).hexdigest()

def slot_ids(row,seat):
    pl=row["players"][seat]
    return list(pl[1])+list(pl[2])

def soft_pick(p,u):
    cs=np.array(p,dtype=np.float64).cumsum()
    return int(np.searchsorted(cs/cs[-1],u,side="right"))

class Uniforms:
    def __init__(self,values):
        self.values=iter(values)
    def choice(self,_,p):
        return soft_pick(p,next(self.values))

def select(row,sampled,model):
    return model._sample(row["players"],row["enemy"],row["lw"],row["lb"],sampled,row["t"])

def read_panel(z,name):
    f=np.load(io.BytesIO(z.read(name)),allow_pickle=False)
    data=json.loads(gzip.decompress(f["meta"].tobytes()))
    y={k:iter(f[k+"_z"]) for k in ["worker","base"]}
    for r in data["rows"]:
        lw=np.zeros((225,5),dtype=np.float32);lb=np.zeros((225,3),dtype=np.float32)
        for _,kind,pos in r["slots"]:
            (lw if kind=="worker" else lb)[pos]=next(y[kind])
        r.update(lw=lw,lb=lb)
    return data

def audit_pair(a,b,model):
    assert a["seed"]==b["seed"] and a["enemy"]==b["enemy"]
    seat=a["enemy"];both=[]
    for side in (1-seat,seat):
        left,right=slot_ids(a,side),slot_ids(b,side)
        common=[u for u in right if u in left]
        moved=[u for u in common if left.index(u)!=right.index(u)]
        both.append(dict(seat=side,common=len(common),shifted=moved,gone=[u for u in left if u not in right],
                         added=[u for u in right if u not in left]))
    left,right=slot_ids(a,seat),slot_ids(b,seat)
    stable=left+[u for u in right if u not in left]
    rng=np.random.RandomState(a["seed"])
    u=rng.random_sample(len(stable))
    res=select(b,Uniforms(u[stable.index(x)] for x in right),model)
    sequential=select(b,Uniforms(u[:len(right)]),model)
    assert sequential==b["teacher"]
    changed=[uid for uid in right if res.get(uid)!=sequential.get(uid)]
    different_choices=[]
    for uid in both[1]["shifted"]:
        if uid in b["players"][seat][2]:
            pos=b["players"][seat][2][uid][0]; probs=model._probs(b["lw"][pos])
        else:
            pos=b["players"][seat][1][uid]; probs=model._probs(b["lb"][pos])
        if soft_pick(probs,u[stable.index(uid)])!=soft_pick(probs,u[right.index(uid)]):
            different_choices.append(uid)
    return dict(terrain_seats=both,root=a["root"],depth=a["depth"],b_hash=b["hash"],a_hash=a["hash"],
                different_draw_choices=different_choices,changed_enemy_orders=changed,
                changed_common_orders=[u for u in changed if u in left],
                changed_new_orders=[u for u in changed if u not in left])

def run(source):
    model=module(source)
    archive=FAMILY/"rollout_readout/simulated_rows.zip"
    results=[]; counts=Counter();entropy=[];per=defaultdict(Counter);totals=Counter()
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not name.endswith(".npz"):continue
            d=read_panel(z,name);rows=d["rows"]
            group=defaultdict(dict)
            for r in rows:
                assert select(r,np.random.RandomState(r["seed"]),model)==r["teacher"]
                model._TEMP=0.
                cand=select(r,np.random.RandomState(r["seed"]),model)
                model._TEMP=1.
                workers=r["players"][r["enemy"]][2]
                out=Counter()
                for w,it in workers.items():
                    p=model._probs(r["lw"][it[0]])
                    h=float(-sum(float(v)*np.log(float(v)) for v in p if v>0))
                    out["workers"]+=1;out["entropy_nats_sum"]+=h;out["one_minus_max_sum"]+=1-float(max(p));out["different_workers"]+=cand.get(w)!=r["teacher"].get(w)
                out["joint_changed"]=cand!=r["teacher"];out["rows"]=1
                per[r["depth"]].update(out)
                group[(r["root"],r["depth"])][r["branch"]]=r
            for pair in group.values():
                if set(pair)!={1,2}:continue
                v=audit_pair(pair[1],pair[2],model)
                results.append(dict(seed=d["seed"],terrain=d["terrain"],source_seat=d["seat"],split=d["split"],**v))
    for r in results:
        counts["sibling_pairs"]+=1
        for idx,k in enumerate(["own","enemy"]):
            s=r["terrain_seats"][idx]
            counts[k+"_common_units"]+=s["common"];counts[k+"_shifted_units"]+=len(s["shifted"])
            counts[k+"_pairs_shifted"]+=bool(s["shifted"]);counts[k+"_pairs_roster_different"]+=bool(s["added"] or s["gone"])
        counts["pairs_either_side_shifted"]+=any(s["shifted"] for s in r["terrain_seats"])
        counts["enemy_proposals_changed_for_shared"]+=len(r["different_draw_choices"])
        counts["enemy_changes"]+=len(r["changed_enemy_orders"])
        counts["enemy_shared_changes"]+=len(r["changed_common_orders"])
        counts["enemy_pairs_changed"]+=bool(r["changed_enemy_orders"])
        counts["enemy_pairs_changed_for_shared"]+=bool(r["changed_common_orders"])
    report=dict(source=sha(source),archive=sha(archive.read_bytes()),counts=counts,
                original_rollout_entropy_greedy_by_depth=per,pairs=results)
    (HERE/"branch_receipts.json.gz").write_bytes(gzip.compress(json.dumps(report,separators=(",",":"),allow_nan=False).encode(),mtime=0))
    print(counts,dict(per),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--original",type=Path);arg=a.parse_args()
    src=arg.original.read_bytes() if arg.original else original()
    assert sha(src)=="65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
    run(src)
