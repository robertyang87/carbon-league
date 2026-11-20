"""Freeze the independent elite-map partition before looking at any root observations."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PANEL=ROOT/"reports/sapling_1200k/master6_actor_dev/broader_elite_backbone_dev/panel.json"
def sha(b): return hashlib.sha256(b).hexdigest()

def main():
    games=json.loads(PANEL.read_bytes())["games"]
    pool=defaultdict(list)
    for g in games:
        if g["split"]=="train" and g["turns"]==299:
            pool[g["teacher"],g["seat"]].append(g)
    selected=[]
    need=lambda v: v or (_ for _ in ()).throw(ValueError("Inconsistent roster"))
    for key,rows in sorted(pool.items()):
        need(len(rows)>=8)
        others=sorted(rows,key=lambda g:sha(("master3s2-own-root-v1|"+g["id"]).encode()))
        versus=[r for r in others if "Master3s2" in r["opponent"]]
        rest=[r for r in others if r not in versus]
        # Keep an original-opponent example in held if the corresponding player/seat has multiple such games.
        train=(versus[:1]+rest)[:6]
        hold=(versus[1:]+[g for g in rest if g not in train])[:2]
        for split,part in (("train",train),("held",hold)):
            for g in part:
                fields={k:g[k] for k in ("id","seed","terrain_sha256","sha256","teacher","opponent","seat","wdl")}
                selected.append({**fields,"split":split})
    need(len(selected)==80 and len({g["seed"] for g in selected})==80 and len({g["terrain_sha256"] for g in selected})==80)
    need(not {g["seed"] for g in selected}&set(range(621758580,621758584)))
    reg=dict(schema="original-master3s2-root-readout/1",original_sha256="65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5",
        source_panel_sha256=sha(PANEL.read_bytes()),games=selected,steps=list(range(7,299,17)),
        method="Only affect original own public-root worker/base readouts. Freeze the 96-channel native trunk, opponent, all rollouts, critic, temperature 1, search wall clock, guards. Teacher-force BeliefTracker every turn with original physical-seat observations and that expert's actual previous order. At 18 fixed turns use each own worker and base's true issued label; ignore source scores, future labels, and postaction board. Include successes and failures; split grouped by whole physical terrain: each of five experts supplies six train/two held per seat. The selection prioritizes at most one already-training game against original Master3s2 per expert/seat, then hash ranks; all other sources come from registered old training pool. No score selection by observation.",
        update="Single convex multinomial calibration of softmax worker and base logits; head equals original deployed (float16-stored) weights plus ridge-regularized delta on frozen original 96 local activations. Freeze once and do not tune on held. Per head each map has equal total loss weight; online inference uses native logits exactly, deploy fp16 effective matrices with zero residual restoring exact original. Standardize RMS per frozen activation over training only (floor 0.25). Optimize 50 fixed L-BFGS iterations, KL(p_original||p_new) coefficient 2.0, L2 coefficient 0.02 on scaled coefficients; bias included. Clip delta by final scalar until mean KL on training <=.01 and 99th percentile abs logit change <=0.7. Both heads combined once. Selection metrics are observational unit orders, not action reward intervention.",
        gate="All 20 held physical maps count; separate worker/base losses, collector/planter role and seats, signs per map. Promote only with overall equal-map mean two-head NLL decrease, both heads non-worsening on average, at least 11/20 maps improve, both seats mean NLL nonworsening, positive and byte-valid zero-delta controls. Mechanism must detect distinct physical orders at original roots under matched random uniforms and show at least one actual change with the full original search plus healthy search work, 0.58s accounting. Development game strength and platform eligibility remain separate. A single scheduled root change is diagnostic when adaptive timing may change search itself.")
    path=HERE/"protocol.json"
    raw=(json.dumps(reg,sort_keys=True,indent=2)+"\n").encode()
    if path.exists() and path.read_bytes()!=raw: raise ValueError("Existing partition differs")
    path.write_bytes(raw)
    print(sha(raw),len(selected))
if __name__=="__main__": main()
