"""Declare the source maps before extracting outcomes or fitting either new outcome head."""
from collections import Counter
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PRE=HERE.parent/"planter_threat_transfer_dev"
sys.path.insert(0,str(PRE))
from registry import read,save,key,sha,TOP

def main():
    panel=read(PRE/"panel.json")["maps"]
    forbidden={r["terrain"] for r in panel}
    nine=read(PRE/"parent_panel/panel.json")["games"]
    forbidden.update(r["map"] for r in nine)
    registered=read(PRE/"splits.json.gz")["maps"]
    pool=[r.copy() for r in registered if r["terrain"] not in forbidden and r["raw_local"]]
    train_ids={r["opponent"] for r in pool if r["split"]=="train"}
    hold_ids={r["opponent"] for r in pool if r["split"]=="new_opponent"}
    assert train_ids.isdisjoint(hold_ids)
    for r in pool:
        if r["split"]=="train" and int(key("two-arm-cal-v1",r["terrain"])[:8],16)%5==0:r["split"]="calibration"
    maps=[]
    for split,n in (("train",80),("calibration",16),("new_opponent",24),("known_new_map",24)):
        groups=Counter();kept=[]
        for r in sorted((r for r in pool if r["split"]==split),key=lambda r:key("two-arm-map-allocation-v1",r["terrain"])):
            if split in ("calibration","known_new_map") and r["opponent"] not in {v["opponent"] for v in maps if v["split"]=="train"}:continue
            if groups[r["opponent"]]>=max(3,n//8):continue
            kept.append(r);groups[r["opponent"]]+=1
            if len(kept)==n:break
        assert len(kept)==n
        maps+=kept
    assert len(maps)==len({r["terrain"] for r in maps})==144
    save(HERE/"plan.json",{"source_split_sha256":sha(PRE/"splits.json.gz"),"old_panel_sha256":sha(PRE/"panel.json"),
        "old_native_panel_sha256":sha(PRE/"parent_panel/panel.json"),"excluded_ladder_panel_maps":len(panel),
        "excluded_native_panel_map_seats":len(nine),"maps":maps,
        "population":"One predetermined Top5 seat per map. Every actual affordable plant/seize location with a submitted focal planter order. Source income ledger joined at time, worker, seat and proposed destination. Fee, receipts and survival are labels only; public history before focal order is the only input besides proposed own simultaneous orders and rules.",
        "target":"For stay: first actual purchase here at t. For moving: first actual purchase here by same worker next turn t+1 at proposed landing spot. Income = official entire first ownership spell gross receipts (including collision cash, excluding double counting enemy cargo) less that purchase's price. Zero when no qualifying purchase. Separate head for conditional stay/move economics; one-step worker loss incurs one publicly quoted planter rehire cost. Score = unconditional expected net_tree minus quoted rehire cost * P(death). Each decision measured independently. One actual tree can appear in a pre-move and purchase label; deduplicate cash effects, never sum these root values or treat them as independent.",
        "model":"Use frozen existing planter_survival learner's public feature construction, no weights. Separate train-only histogram tree gradient ensembles for unconditional net and next-turn death. Calibration from calibration maps only with fixed affine/ridge calibration grouped by stay/move, death uses logistic-on-logits with shrink and clipping. Conservative tie-break returns move (parent); no tuned threshold. Baseline train action means, geometry/last-direction fixed learner available for comparison.",
        "test":"Held map and separate whole-registration opponents: map-equal death Brier/logloss and net MAE; paired same historical committed enemy first step on exact Sapling A roots, net survival and actual purchases. Both-arm continuation required for direct official terminal claim; 64 previously studied native archive roots reported as diagnostics by map and all roots. No new games unless later separately approved.",
        "gate":"Stop before new original games if either external held split fails to improve both Brier and net MAE over trained stay/move constants, or unknown-opponent paired A roots have negative net worker survival vs A, or archived paired actual terminal margin vs parent not positive on at least half available physical maps and overall. Measured calibration and one-step impact kept separate from attribution of true score."})
    print(Counter(r["split"] for r in maps))
if __name__=="__main__":main()
