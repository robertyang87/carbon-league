"""Freeze six original-opponent training boards and exclude them permanently from independent evaluation."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
LEAGUE = HERE.parent
REPEAT = LEAGUE/"natural_original/fresh_repeat"
spec = importlib.util.spec_from_file_location("repeat_clearance", REPEAT/"clearance.py")
repeat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repeat)
REG=repeat.registry
SNAPSHOT="87688aac9aaf5d0bf5c9f59b420617a15a2bf850"
PREVIOUS="40ddc60e05205c1ac2e79f66044fa9369d4cca6e"
SEEDS=list(range(621758460,621758466))

def history():
    REG.SNAPSHOT=SNAPSHOT
    REG.GROUPS={"original_on_policy_training":SEEDS}
    REG.SEEDS=SEEDS
    REG.CANDIDATE=re.compile(rb'(?<![.\d])62175846[0-5](?![.\d])')
    census=REG.scan()
    census.update(snapshot=SNAPSHOT,source_snapshot=SNAPSHOT,seeds=SEEDS,
                  purpose="Six new training-only boards for natural on-policy rollouts against original Master3s2. All six seeds AND physical layouts are excluded from independent development and the final examination permanently.")
    return census

def recent(cache):
    # The previous ledger paid for the official replay of every recently added score reference.
    # Only the subsequent score increment may use public retrieval.
    repeat.SCORE_BASE=PREVIOUS
    repeat.SNAPSHOT=SNAPSHOT
    return repeat.recent(cache)

def frozen_bytes(p):
    return REG.git("show",SNAPSHOT+":"+p.relative_to(REG.ROOT).as_posix())

def merge(h,r):
    if h["source_snapshot"]!=SNAPSHOT or r["end"]!=SNAPSHOT or h["seeds"]!=SEEDS:raise ValueError("Source boundary mismatch")
    boards=set(h["by_seed"].values());seeds=set(SEEDS)
    def check(pairs,group):
        for s,b in pairs:
            if not isinstance(s,int) or len(b)!=64 or not re.fullmatch("[0-9a-f]{64}",b):raise ValueError("Broken map identity: "+group)
            if s in seeds or b in boards:raise ValueError("Previously used or reserved: "+group)
    def bind(p):
        data=frozen_bytes(p);return json.loads(data),hashlib.sha256(data).hexdigest()
    previous,last_clear=bind(REPEAT/"clearance.json")
    evidence,last_increment=bind(REPEAT/"pre_dispatch_increment.json")
    if previous["source_snapshot"]!=evidence["from_sha"] or evidence["through_sha"]!=PREVIOUS or evidence["frozen_clearance_sha256"]!=last_clear:
        raise ValueError("Broken prior snapshot chain")
    repeat_plan,repeat_plan_hash=bind(REPEAT/"protocol.json")
    if repeat_plan["expected_games"]!=8 or set(repeat_plan["seeds"])!={621758450,621758451} or repeat_plan["physical_boards"]!=previous["by_seed"]:
        raise ValueError("Original repeat's registered physical maps differ")
    f59,f59_hash=bind(LEAGUE/"native_gradient/balanced_step/original_response/protocol.json")
    if f59["student"]!="f59f7d2559383ac8835e0563a52ecc2bc5fc48c27bd049ebd7f7842e9a35fe4f": raise ValueError("Wrong additional historical model")
    check(((v["seed"],v["terrain"]) for v in f59["cases"]),"f59 historical root continuations")
    for group, known in f59["training_physical_maps"].items():
        check(((int(s),b) for s,b in known.items()),"f59 fitted maps "+group)
    histories=previous["latest_score_delta"]["verified_replays"]
    check(((v["seed"],v["opening_225_u8_sha256"]) for v in histories),"previous official replay census")
    previous_original,previous_registration=bind(LEAGUE/"clearance.json")
    check(((int(s),b) for s,b in previous_original['by_seed'].items()),"previous 26-board training and reservations")
    check(((int(s),b) for s,b in previous['by_seed'].items()),"450/451 two-map original repeat")
    check(((v["seed"],v["layout_sha256"]) for v in evidence["new_archived_ladder_replays"]),"later original replay blobs")
    check(((v["seed"],v["opening_225_u8_sha256"]) for v in r["verified_replays"]),"new ladder score references")
    if {v["game"] for v in r["verified_replays"]}!=set(r["compared_ids_including_mirrors"]):raise ValueError("Unverified additional score reference")
    paths=[p for p in REG.git("ls-tree","-r","--name-only",SNAPSHOT,str(repeat.TOP.relative_to(REG.ROOT))).decode().splitlines() if p.endswith("/validation_maps.jsonl.gz")]
    archived=[json.loads(line) for p in paths for line in gzip.decompress(REG.git("show",SNAPSHOT+":"+p)).splitlines()]
    new_top=set(r["first_seen_top5"])
    active_rows=[v for v in r["verified_replays"] if v["game"] in new_top]
    terrains={v["initial_terrain_u8_sha256"] for v in archived}; ids={v["id"] for v in archived}
    if len(ids)!=h["top5_unique_openings"] or new_top&ids or set(v["game"] for v in active_rows)!=new_top:raise ValueError("Top-five increment not distinct or complete")
    terrains.update(v["opening_225_u8_sha256"] for v in active_rows)
    if any(h["formal_seed_half_open"][0]<=s<h["formal_seed_half_open"][1] for s in seeds) or len(boards)!=6:
        raise ValueError("Reserved or duplicated training map")
    return {**h,"by_seed":h["by_seed"],"train_seeds":SEEDS,"snapshot":SNAPSHOT,"source_snapshot":SNAPSHOT,
            "permanently_excluded_from_independent_evaluation":{"seeds":SEEDS,"physical_boards":h["by_seed"]},
            "previous_actual_release":{"prior_repeat_clearance_sha256":last_clear,"prior_dispatch_increment_sha256":last_increment,
            "existing_26_map_clearance_sha256":previous_registration,
            "eight_game_repeat_plan_sha256":repeat_plan_hash,"f59_original_protocol_sha256":f59_hash,
            "f59_registered_original_cases":[{k:v[k] for k in ("seed","seat","terrain")} for v in f59["cases"]],
            "f59_fit_maps_by_group":{k:len(v) for k,v in f59["training_physical_maps"].items()},"physical_450_451":previous["by_seed"],
            "reserved_old_greedy_444_445":{str(s):previous_original["by_seed"][str(s)] for s in (621758444,621758445)},
            "last_verified_ladder_score_references":len(histories),"post_clearance_archived_replays":len(evidence["new_archived_ladder_replays"])},
            "top5_archived_physical_registry_rows":len(archived),"top5_new_scored_game_ids":sorted(new_top),
            "top5_games_scored":h["top5_games_scored"]+len(active_rows),"top5_unique_openings":len(terrains),
            "top5_total_distinct_game_ids_including_faults":len(ids)+len(active_rows),
            "new_score_delta":r,"physical_intersections":[],"seed_intersections":[],
            "coverage":"Archived physical openings and registered identities through the frozen Git revision, including the full Top Five physical registry plus later score-index additions verified from public replays. Tests against archived terrains use actual initial cells; metadata-derived synthetic local boards are a conservative additional exclusion. Unpublished private games cannot be enumerated."}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch",required=True,type=Path);parser.add_argument("--stage",choices=("all","census","recent","finish"),default="all")
    a=parser.parse_args();a.scratch.mkdir(exist_ok=True,parents=True)
    if a.stage in ("all","census"):repeat.freeze(a.scratch/"census.json",history())
    if a.stage in ("all","recent"):repeat.freeze(a.scratch/"recent.json",recent(a.scratch/"replays"))
    if a.stage in ("all","finish"):
        result=merge(json.loads((a.scratch/"census.json").read_bytes()),json.loads((a.scratch/"recent.json").read_bytes()))
        repeat.freeze(HERE/"clearance.json",result)
if __name__=="__main__":main()
