"""Read sealed Top Five replays once for a fixed out-of-sample policy comparison."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"code"))
import numpy as np
import torch
from cnet2_model import CNet2
from sapling_full_pass import score
from cnet2_build import build_game


def checkpoint(path):
    with np.load(path,allow_pickle=False) as f:
        net=CNet2(**json.loads(str(f["meta"]))["cfg"])
        net.load_state_dict({k:torch.from_numpy(f[k]) for k in f.files if k!="meta"},strict=True)
    return net.eval()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    os.environ["BC_DEVICE"]="cpu"
    import cnet2_train as trainer
    torch.set_num_threads(1)
    frozen=json.loads((HERE/"top5_h2h_data.json").read_text())
    names=set(frozen["strategy_ids"])
    known={g["seed"] for g in frozen["games"]}
    weights=ROOT/"data/weights/sapling_1200k"
    paths=[weights/"cash_v3.npz",weights/"full_top5_decoder.npz"]
    nets=list(map(checkpoint,paths))
    inventory_bytes=args.inventory.read_bytes()
    inventory=json.loads(inventory_bytes)
    rows=[]
    excluded=[]
    for i,g in enumerate(inventory["games"]):
        if not g.get("eligible") or g.get("seed") in known:
            excluded.append(g["id"])
            continue
        file=args.cache/g["path"]
        if hashlib.sha256(file.read_bytes()).hexdigest()!=g["sha256"]:
            raise ValueError("sealed replay hash mismatch: "+str(file))
        replay=json.loads(gzip.decompress(file.read_bytes()))
        T=len(replay["steps"])-1
        sides=[s for s,name in enumerate(g["seat_names"]) if name in names]
        if not sides: continue
        if replay["statuses"]!=["DONE","DONE"] or len(replay["rewards"])!=2:
            raise ValueError("incomplete replay")
        with tempfile.TemporaryDirectory() as tmp:
            shard=Path(tmp)/"states.npz"
            margins=[replay["rewards"][s]-replay["rewards"][1-s] for s in sides]
            _,n,error=build_game((str(file),sides,i,[2]*len(sides),margins,str(shard)))
            if error or n!=len(sides)*T: raise ValueError([g["id"],n,error])
            with np.load(shard,allow_pickle=False) as archive:
                data={k:archive[k] for k in trainer.FIELDS}
            results={}
            for index,s in enumerate(sides):
                ix=np.arange(index*T,(index+1)*T)
                results[str(s)]={path.stem:score(net,data,ix,trainer) for net,path in zip(nets,paths)}
            rows.append(dict(id=g["id"],seed=g["seed"],cohort=g["cohort"],seat_names=g["seat_names"],
                             transitions=T,packed_sha256=g["sha256"],by_seat=results))
    if not rows: raise ValueError("no new, sealed and eligible replay")
    result=dict(checkpoint_sha256={p.stem:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                inventory_sha256=hashlib.sha256(inventory_bytes).hexdigest(),
                excluded=excluded,games=rows,policy_style=2,
                units="All actionable slots with a registered Top Five seat; prefix uses the observed prior commands. Games include opposing seats only if both were Top Five. Unweighted conditional action likelihood; never a win-rate estimate.")
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
