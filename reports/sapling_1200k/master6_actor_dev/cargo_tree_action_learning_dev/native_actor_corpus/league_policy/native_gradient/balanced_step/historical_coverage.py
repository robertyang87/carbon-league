"""Measure post-freeze physical action coverage on the prior paired original forks."""
from collections import Counter, defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import argparse
import hashlib
import json
import os

HERE=Path(__file__).resolve().parent
LEAGUE=HERE.parents[1]
TRANSFER=LEAGUE/"transfer_probe.py"
CHILD="f59f7d2559383ac8835e0563a52ecc2bc5fc48c27bd049ebd7f7842e9a35fe4f"
PARENT="8da4b465f7fa6e601e51bd7c2e81f25825cb59b9b5ab285f7f56d0f6bbfe1cac"

def load(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def check(ok,why):
    if not ok:raise ValueError(why)

def frozen_training():
    plan=load(HERE/"protocol.json")
    archive=HERE.parent/"alignment.json"; tickets=LEAGUE/"collection.json"; clearance=LEAGUE/"clearance.json"
    check(plan["parent_sha256"]==PARENT and plan["historical_alignment_sha256"]==sha(archive)
          and plan["league_collection_sha256"]==sha(tickets),"Training cohort provenance changed")
    historic=load(archive)["original_inputs"]["games"];train=load(tickets)["games"]
    def identities(records,expected):
        seeds=defaultdict(set);seats=defaultdict(set)
        for r in records:
            seeds[r["seed"]].add(r["terrain"])
            seats[r["seed"]].add(r["seat"])
        check(len(seeds)==expected and all(len(v)==1 for v in seeds.values()) and all(v=={0,1} for v in seats.values()),"Missing seed/seat or ambiguous terrain")
        return {s:next(iter(v)) for s,v in seeds.items()}
    a=identities(historic,4); b=identities(train,18)
    check(set(a)=={621758345,621758346,621758348,621758349} and set(b)==set(load(clearance)["groups"]["train"]),"Wrong training cohorts")
    check(all(load(clearance)["by_seed"][str(s)]==v for s,v in b.items()),"League terrain differs")
    return a,b

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--inventory",action="store_true");args=ap.parse_args()
    original,league=frozen_training()
    spec=spec_from_file_location("fixed_original_pairs",TRANSFER);probe=module_from_spec(spec);spec.loader.exec_module(probe)
    rows=probe.cases()
    keep=[];exclude=[]
    for r in rows:
        matches={k:sorted(s for s,h in train.items() if h==r["terrain"] or s==r["seed"])
                 for k,train in (("original",original),("league",league))}
        if any(matches.values()):
            exclude.append(dict(case=r["case"],seed=r["seed"],terrain=r["terrain"],seat=r["seat"],t=r["t"],overlap=matches))
        else:keep.append(r)
    check(len(keep)+len(exclude)==16,"Incomplete original comparison registry")
    print("Original registered / eligible / excluded",len(rows),len(keep),len(exclude),flush=True)
    if args.inventory:return
    child=HERE/"candidate.py"
    check(sha(child)==CHILD and sha(probe.EXP/"candidate.py")==PARENT,"Unfrozen model")
    measured=probe.measure(keep,child)
    check(len(measured)==len(keep),"Incomplete public replay")
    counts=Counter(x["status"] for x in measured)
    supported=[r for r in measured if r["status"]=="supported_change"]
    check(all(x["candidate_official_margin"] is None for x in measured if x["status"]=="unmeasured_change"),"Invented missing outcome")
    for r in measured:
        r["official_wdl_if_observed"]={
            arm:None if r[f"{arm}_official_margin"] is None else int(r[f"{arm}_official_margin"]>0)-int(r[f"{arm}_official_margin"]<0)
            for arm in ("parent","candidate")}
    signs=Counter("positive" if r["delta_official_margin"]>0 else "negative" if r["delta_official_margin"]<0 else "zero" for r in supported)
    training=load(HERE/"protocol.json")
    out=dict(schema="single-balanced-update-historical-action-coverage",parent_sha256=PARENT,candidate_sha256=CHILD,
        program_sha256=sha(Path(__file__)),transfer_code_sha256=sha(TRANSFER),training_plan_sha256=sha(HERE/"protocol.json"),
        retired_original_training_by_seed=original,league_training_by_seed=league,
        source_manifest_sha256=training["historical_alignment_sha256"],source_game_register_sha256=training["league_collection_sha256"],
        late_early_source_digests={str(k.relative_to(probe.ROOT)):v for k,v in probe.INPUTS.items()},
        registered=16,eligible=len(keep),excluded=exclude,eligible_physical_terrains=len({r["terrain"] for r in keep}),
        outcomes={key:counts[key] for key in ("unchanged","supported_change","unmeasured_change")},
        supported_changes_by_sign=dict(signs), supported_change_physical_terrains=len({r["terrain"] for r in supported}),
        rows=measured,scope="Existing conditional single-action experiments. A new joint order absent from the same recorded fork is unknown. An identical order has zero change on the same stored draw and common continuation, so it never counts as evidence for a treatment. Rewards are not aggregate full-game candidate results. No newer full-game or separate teacher-window outcomes are read.")
    path=HERE/"coverage.json";tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n");os.replace(tmp,path)
    print(out["outcomes"],out["supported_changes_by_sign"],sha(path),flush=True)
if __name__=="__main__":main()
