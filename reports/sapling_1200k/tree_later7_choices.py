"""Check the learned planting action under the same logged Top Five prefixes as the defense audit."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import h2h_late_holdout as holdout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args=p.parse_args()
    if any(k.startswith(("BC_", "CNET2_")) for k in os.environ):
        raise ValueError("unset training overrides")
    os.environ["BC_DEVICE"]="cpu"
    torch.set_num_threads(1)
    import cnet2_train as trainer
    rows, groups, _, _, games, _, _ = holdout.collect(args.raw_dir)
    selections=[]
    for r in rows:
        own=holdout.F.OWN
        planter=r["tok_kind"][:own]==1
        features=r["mf"]
        stay=features[:,0]
        vacant=~stay[:,[5,6,8,9]].any(1)
        affordable=r["scal"][13]>0
        opponents=r["tok_pos"][12:22][r["tok_mask"][12:22] & (r["tok_kind"][12:22]==4)]
        near=(holdout.F.DIST[r["tok_pos"][:own,None],opponents[None,:]]<=2).any(axis=1)
        plantable=planter & vacant & affordable
        selections.append(np.stack([plantable & near,plantable & ~near,planter & (stay[:,9]>0)],axis=-1))
    eligible=np.stack(selections)
    # score accepts three arbitrary eligibility masks and a response lookup per legal command.
    # Its internal context names correspond to mask positions and are renamed in the receipt.
    response=np.zeros((*eligible.shape,5),bool)
    response[:,:,:,0]=True
    names=("open_affordable_enemy_planter_within_two","open_affordable_other","occupying_enemy_tree")
    summary={}
    for model, fname in (("v3","cash_v3.npz"),("v4","tree_v4.npz")):
        stats=holdout.score(holdout.WEIGHTS/fname,rows,groups,eligible,response,trainer)
        counts={n:stats["contexts"][k] for n,k in zip(names,holdout.CONTEXTS)}
        summary[model]=dict(weight_sha256=stats["sha256"],counts=counts)
    report=dict(roots=len(rows),plays=[g["id"] for g in games],ticks=holdout.TICKS,models=summary,
                evaluator_sha256=holdout.digest(Path(__file__).read_bytes()),
                scope="Frozen logged-prefix FP32 choices; public features only, expert_yes and policy_yes count STAY. "
                "Open cell excludes either base and any tree. Affordability uses the public root cash and current tree price. "
                "Enemy-tree occupancy is shown without an affordability condition. Realized plant or capture requires a simulator transition.")
    with args.out.open("x") as f:
        json.dump(report,f,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps(summary))


if __name__=="__main__":
    main()
