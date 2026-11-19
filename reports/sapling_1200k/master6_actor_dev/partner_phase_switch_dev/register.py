"""Freeze source eligibility, budget, predeclared turn and outcome/gates before making native labels."""
import json,hashlib,gzip
from pathlib import Path
from collections import Counter

HERE=Path(__file__).resolve().parent;ACTOR=HERE.parent;BASE=ACTOR.parent;ROOT=BASE.parents[1]
PARTNER=ACTOR/"actor_partner_dev";INDEX=ACTOR/"stratified_action_value_dev/source/inventory.json"
def read(p):
    b=p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix==".gz" else b)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stable(*x):return hashlib.sha256(json.dumps(x,separators=(",",":")).encode()).hexdigest()
def seal(p,v):
    if p.exists():
        if read(p)!=v:raise ValueError(f"frozen receipt differs {p}")
        return
    b=(json.dumps(v,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n").encode()
    tmp=p.with_suffix(".partial");tmp.write_bytes(gzip.compress(b,mtime=0) if p.suffix=='.gz' else b);tmp.replace(p)

def main():
    old=read(PARTNER/"protocol.json");new=read(PARTNER/"replication/protocol.json")
    blocked=set(old["maps"])|set(new["new_dev"]["seeds"])
    inv=read(INDEX);rows=[]
    for row in inv["v1_inventory"]:
        if row["seed"] in blocked:continue
        if row["pack_sha256"]!=new["parent_pack_sha256"] or row["steps"]<226:continue
        subset=int(stable("partner-phase-map-split",row["map"])[:8],16)%3
        rows.append({k:row[k] for k in ("seed","seat","map","raw","raw_sha256","steps")}|{"split":"held" if subset==0 else "train"})
    assert len(rows)==len({(r["map"],r["seat"]) for r in rows})
    seal(HERE/"protocol.json",{"source_inventory_sha256":sha(INDEX),"frozen_policy_sha256":sha(HERE/"policy.py"),
        "previous_partner_protocol_sha256":sha(PARTNER/"protocol.json"),"new_partner_protocol_sha256":sha(PARTNER/"replication/protocol.json"),
        "eligible_source_maps":len({r["map"] for r in rows}), "excluded_partner_evaluation_maps":len(blocked),"eligible":rows,
        "partners":{"it50":new["parent_pack_sha256"],"it20":new["candidate_pack_sha256"],"examiner":new["original_sha256"]},
        "selection":"Eligible prior original-v1 trajectories only, no old/new full-policy partner evaluation map. Assign split by terrain SHA before observing. At fixed t=195 or225 evaluate public regime using last 10 observations; sample with fixed hash independent of commands or results. Three source maps per regime for train and two other maps per regime for held. Up to one root per physical map; fixed both-seat and both-time quotas via deterministic minimum-hash combination. Do not select on partner disagreement, margin, WDL, actor preference or seed quality. If infeasible, stop.",
        "action":"Each learned decision commits from the designated current public history onward, retaining Sapling 25% at every step. it20 uses Echo4's packed actor, it50 uses the old PPO checkpoint (both 75% of mixture); source prefixes all physically executed old mixed it50. Hold neither slot then re-propose nor single-command intervention. Freeze a hypothetical deploy turn at t195 or t225 per schedule; physical map/seat/identity cannot determine which partner. Training effects conditional on switching at these root times, not an opening policy comparison.",
        "features":"Cargo collectable near base vs contest by lighter enemy workers, relative paid capacity (collectors, tree age), 10-step cash-flow disadvantage. Pressure iff at least two fixed flags from policy.py, else stable. No seat number, seed or later data as predictor.",
        "support":"Ten physical maps available, 6 training (3/3 regimes across 3 seat0, 3 seat1 and 3 t195 and 3 t225); four held (2/2 across 2 seat0/2 seat1 and 2 each time), each subset includes both states at both times and seats if feasible. On respective true public history compare actual A/B open-entry joint. Before search require >=2 train and >=1 held map with changed order in retrospectively walked 3-step public root vicinity, and Sapling-ablation changes evidence pooled across both split histories. Root difference not required, as sustained policy may diverge next step.",
        "train":"Map equal unit, one primary stream per map. Two conditional means; select it20 only if signed mean official terminal delta >0 and at least two empirical positives within that regime; else it50. Baselines always it50 and always it20 at the same root. Leave one training map out for each prediction; both partners selected across roots, positive mean vs each constant before moving to held. Repeat FIRST selected training root from each regime using a second independent common random stream; if both primary and duplicate delta reverse sign in both preregistered cases, stop. Record sign changes individually, no pooling repeats as maps; complete all signed training and repeat cases before verdict. No held responses after failed gate.",
        "held_gate":"After passing training gate lock learned mapping; same 4 untouched physical source maps. Compare to both constant partners and underlying it50; require net positive official margin vs both controls with no fewer wins than either. Show each map and seat separately and bootstrap by map; historical parent terminal is not a paired response. These small conditional continuations do not measure full-policy wins, and no formal maps used.",
        "budgets":{"active_seconds":2100,"max_primary_train_pairs":6,"max_additional_training_pairs":2,"max_held_pairs":4,"native_workers":1,
            "root_times":[195,225],"max_native_arms":24,"maximum_from_source_measurement":"Previous 1092 original follow-up calls in 748.86 active seconds. New plan roughly 2000 follow-up calls; wall time not guaranteed. Active budget shared across all stages and restarts. Waiting for the one original court after current registered eight games is free. Stop on insufficient budget, never drop hard roots then call it a pass."}})
    print(Counter(r["split"] for r in rows))
if __name__=="__main__":main()
