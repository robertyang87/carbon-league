import json, hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
TR=HERE.parent/"tree_response_training"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def register():
    sources=[TR/"source_receipt.json"]+[TR/f"batches/intake_0{i}/source_receipt.json" for i in range(2,6)]
    ranges={}
    boards=set()
    for k,p in enumerate(sources,1):
        r=json.loads(p.read_bytes()); key="train" if k<=3 else "held_maps"
        for g in r["games"]:
            if (g["seed"],g["seat"]) in boards:raise ValueError("duplicate seat")
            boards.add((g["seed"],g["seat"]))
        ranges.setdefault(key,[]).append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),maps=sorted(set((g["seed"],g["map_sha256"]) for g in r["games"]))))
    boardsets=[{h for batch in rows for seed,h in batch["maps"]} for rows in ranges.values()]
    if [len(bs) for bs in boardsets]!=[6,4] or not set.isdisjoint(*boardsets) or len(boards)!=20:raise ValueError("map split")
    obj=dict(splits=ranges,agent_sha256="9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0",
        scope="All positive-cargo own collector decisions on 10 source maps against original Master3s2; fit six maps (01,02,03), hold out four (04,05). Full-map split, both physical seats together. Previously analyzed maps, independent of this fit; 3 prior examples belong to held 04 and will be flagged, never tune on them.",
        mechanism="Ensemble of 16 randomized regression trees (fixed seed 11962026, max depth 8, min leaf 60, feature fraction 0.75, no bootstrapping). Equal total importance per source map. Limit BLAS/estimator to a single CPU. Train two views per positive root in one model: public state-only view with explicit unknown first action, and actionable view with the actual first v1 order. Include all eligible workers and both seats. Existing 10-step estimator retained for comparison. Second same-family baseline sees only cargo, distance and remaining time. No tuning from held maps or external branches.",
        outputs="Fractional competing terminal causes, net cash divided by initial cargo; cash mass and time first moment per payment time bin [1,3,6,10,16,25,40,64,100,terminal]. A unit of money still carried at day 10 is still at risk; complete tape follows it, including same-team collision transfer and future cash. Opponent acquisition is a terminal loss of tagged principal, subsequent reacquisition is a separate economic event. Territory and new harvest are excluded. Real rule termination counts remaining inventory as not cashed. Reject truncated logs; do not mark them failed deliveries. Positive roots with no future cash have no conditional cash-return time.",
        evaluation="Factual per-map turn/worker counts, cash forecast error (amount and normalized terminal recovery, 10-step and cumulative), cash-weighted time error among real receipts, calibration of losses, 10-step right-censored still-at-risk inventory and its eventual disposition, macro map comparisons. Old head's existing 10-turn cash prediction credited as 10-turn; quantify bias of reusing it as an eventual-cash proxy. Independent pre-existing matched native alternative-action roots checked on different physical maps from all 10 sources, predictive paired differences and number of non-ties. No new native games or official registered exam. Treat an action association as observational until true new-map paired responses show real incremental economic value. No veto ships solely on forecast error improvement.",
        public_input="Current same-seat public observation, at most 5 earlier same-seat observation snapshots, first focal order optionally and actual contemporaneous friendly orders only. Never consult opponent current order, future position, simulator, full carbon, seed, terminal scores, map ID or opponent name to predict. Unit ID used solely to match earlier states/entities.")
    path=HERE/"protocol.json"; b=(json.dumps(obj,indent=2)+"\n").encode()
    if path.exists() and path.read_bytes()!=b:raise ValueError("frozen spec")
    path.write_bytes(b)
if __name__=="__main__":register()
