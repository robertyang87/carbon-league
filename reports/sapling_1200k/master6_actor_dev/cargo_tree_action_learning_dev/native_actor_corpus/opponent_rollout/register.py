"""Freeze the natural-search demonstrations and evaluation roots without reading future outcomes."""
from collections import defaultdict
from pathlib import Path
import gzip
import hashlib
import json

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent
CARGO = CORPUS.parent
ACTOR = CARGO.parent
ROOT = ACTOR.parents[2]
OLD = ACTOR / "own_search_bridge_dev/opponent_response_dev"
EXACT = {621758345, 621758346, 621758348, 621758349}
PARENT = CARGO / "candidate.py"
SEED = 635211
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):
    return str(p.relative_to(ROOT))
def load(p):
    b = p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix == ".gz" else b)
def save(p, obj):
    data = json.dumps(obj, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=1).encode()+b"\n"
    p.write_bytes(gzip.compress(data,mtime=0) if p.suffix==".gz" else data)
def rowcard(r):
    return {k:r[k] for k in ("seed","expert_seat","candidate_seat","opponent_pack_sha256","raw_sha256","physical_sha256","path","transitions")}

def register():
    inventory = CORPUS/"expansion/inventory.jsonl.gz"
    if sha(inventory)!="3220787149a5d54b8638d049897e598b0204734c97a44f927b6046fa73de6ae2":
        raise ValueError("frozen expansion tape index differs")
    previous=load(CORPUS/"sources.json")["cards"]
    oldsplit={r["sha256"]:r["split"] for r in previous}
    rows=[json.loads(l) for l in gzip.open(inventory)]
    groups=defaultdict(list)
    for r in rows:
        if not r["native_action_supervision"] or not r["native_outcome_supervision"]:
            continue
        seed=r["seed"]
        if seed in EXACT:
            if r["opponent_pack_sha256"]==sha(PARENT):
                groups["exact"].append(rowcard(r))
        elif r["sampling_pool"]=="protected":
            split=oldsplit.get(r["raw_sha256"])
            if split in ("val","test"):
                groups[split].append(rowcard(r))
        elif r["sampling_pool"] in {"current_13_enrolled","current_13_extra_trajectory","strict_new","retirement_outside_29","retirement_quarantine_7"}:
            groups["train"].append(rowcard(r))
    for split, expected in dict(train=(63,260),val=(3,6),test=(6,12),exact=(4,8)).items():
        group=groups[split];assert (len(set(r["physical_sha256"]for r in group)),len(group))==expected
    terrains={k:{r["physical_sha256"]for r in v} for k,v in groups.items()}
    assert sum(map(len,terrains.values()))==len(set.union(*terrains.values()))==76
    assert sha(PARENT)=="8da4b465f7fa6e601e51bd7c2e81f25825cb59b9b5ab285f7f56d0f6bbfe1cac"
    for k,v in groups.items():
        assert all(r["expert_seat"]==1-r["candidate_seat"] for r in v)
        assert all(r["seed"] not in (621758400,621758401) for r in v)
        v.sort(key=lambda r:(r["seed"],r["expert_seat"],r["raw_sha256"]))
    sources=HERE/"sources.json.gz";save(sources,dict(groups))
    files=[inventory,CORPUS/"expansion/exclusions.json",CORPUS/"sources.json",PARENT,OLD/"opponent.npz",
           OLD/"active_calibration/opponent.npz",OLD/"train.py",OLD/"sources.json",OLD.parent/"bridge.py",
           ROOT/"code/cnet2_feat.py",ROOT/"code/fastsim.py",HERE/"register.py",sources]
    plan={"frozen_input_revision":"b98e61c79","input_sha256":{rel(p):sha(p) for p in files},
          "seeds_by_split":{k:sorted(set(r["seed"]for r in v))for k,v in groups.items()},
          "record_count":{k:len(v)for k,v in groups.items()},"seed":SEED,
          "model":"Independent 83-input/48 tanh-hidden scalar action scorer, 4080 learnable parameters. No original policy code or weights. Reuse public cnet2_feat and bridge action features, physical expert viewpoint.",
          "train":"Retire prior development observations for this version. Take 64 distinct real roots per terrain and expert seat across its eligible trajectories, allocating as evenly as possible across complete matches then their time segments. Follow observation history from t0. Use the actually executed same-turn expert commands only as previous-slot prefix in fitting. Report first unrepresentable prefix and dropped suffix. Fit all periods. Fixed 12 Adam epochs, lr=.002 batch 384, cross-entropy on nonforced representable slots; no diagnostic-based tuning.",
          "validation":"Three original validation maps, fixed 64 stratified turns per map/seat. Choose 1 scalar softmax temperature from [0.70,0.85,1,1.2,1.5] by map-equal action NLL on nonforced representable slots, with self-predicted greedy prefixes. Argmax is unchanged.",
          "diagnostic":"Six original diagnostic maps, original registered two seats per map, 64 held turns per tape. Score each free-running unit prefix, physical role and seat. Report NLL, Brier, choice support, active/loaded collector actions and joint match. Old baseline previously fit one of these six, 621495120. New never fits any of the nine. Four separate archived 8da4 maps withheld in full from all estimation.",
          "baselines":["Published original-examiner opponent_response_dev/opponent.npz (3400 parameters, six prior fitting maps, primary)","Published active_calibration/opponent.npz (3485 parameters, secondary)"],
          "closed":"Each of four exact-lineup held maps and six general diagnostic maps, both expert seats, start t=70,175, settle each arm after 5 and 10 new joint decisions. 8da4 calls its normal entry on its own branch's observations for every simulated step. Simulated opponent sees its own causal physical observation plus history. All 3 arms use the same reconstructed initial physical state. Archive only defines historical truth, never future actor replies. General 6 maps actually faced 9c54df17, report separately. Report deposits, cash, tree ownership and matched-wall throughput, age-independent map-equal MAE. No original examiner invoked.",
          "interpretation":"New is promising for future cheap screening only if on the 4 exact maps mean absolute error vs archived reference falls against primary at H10 for both expert deposits and end tree-count, H5 has no increase on either measure, and free held NLL improves. Otherwise report mixed/negative. Historical conditional evidence only. Full-information root is for offline auditing. Future search inside a deployed Sapling must build the opponent view from the Sapling public belief world.",
          "budget":"CPU 3 one numerical thread, no new games, residual-cache creation in /tmp only; cap 1,200 CPU seconds ETL/fit/held and 600 CPU seconds closed evaluation. If exceeded, report unfinished portion rather than select a different result. Fixed trained checkpoint regardless of results."}
    dest=HERE/"protocol.json"
    if dest.exists() and load(dest)!=plan:raise ValueError("contract already frozen")
    save(dest,plan);print(sha(dest),plan["record_count"],flush=True)
if __name__=="__main__": register()
