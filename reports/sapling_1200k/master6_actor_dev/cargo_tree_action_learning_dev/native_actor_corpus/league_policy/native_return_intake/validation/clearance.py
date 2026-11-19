"""Freeze two independent development boards strictly outside the ongoing original-opponent training intake."""
import argparse
import gzip
import hashlib
import importlib.util
import io
import tarfile
import zipfile
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
INTAKE=HERE.parent
LEAGUE=INTAKE.parent
spec=importlib.util.spec_from_file_location("native_intake_clearance",INTAKE/"clearance.py")
prior=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
REG=prior.REG
REP=prior.repeat
CENSUS="9e1e18641781a6b6ac8d3f86cbe057af49111ba7"
SOURCE="f564ae414be7672db93cb1033bc2af75c2d88142"
PRIOR="87688aac9aaf5d0bf5c9f59b420617a15a2bf850"
SEEDS=[621758470,621758471]
REGISTRY_SHA="f720c89c4d221f43f3cec81a73a3ce765c4d70c14dae44ee76c289a04ad346fc"

def require(ok,message):
    if not ok:raise ValueError(message)

def bound(path):
    data=REG.git("show",SOURCE+":"+path.relative_to(REG.ROOT).as_posix())
    return json.loads(data),hashlib.sha256(data).hexdigest()

def census():
    REG.SNAPSHOT=CENSUS
    REG.SEEDS=SEEDS
    REG.GROUPS={"held_native_development":SEEDS}
    REG.CANDIDATE=re.compile(rb"(?<![.\d])62175847[01](?![.\d])")
    return REG.scan()

def score_delta(scratch):
    REP.SCORE_BASE=PRIOR
    REP.SNAPSHOT=SOURCE
    return REP.recent(scratch/"replays")

def changed_evidence(boards):
    # The old complete scanner explicitly locks the 1,637-entry historical census.
    # Inspect every changed blob and nested member since that census, including the two new full replays.
    patterns=[str(s).encode() for s in SEEDS]+[v.encode() for v in boards.values()]
    seed_pattern=re.compile(rb"(?<![.\d])62175847[01](?![.\d])")
    paths=REG.git("diff","--name-only","-z",CENSUS,SOURCE).split(b"\0")
    manifest={};replay_openings={}
    def inspect(path,raw):
        if path.endswith(".gz"):
            return inspect(path[:-3],gzip.decompress(raw))
        if path.endswith(".tar"):
            with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
                for item in archive:
                    if item.isfile():inspect(path+"/"+item.name,archive.extractfile(item).read())
            return
        if path.endswith((".npz",".zip")):
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                for member in archive.namelist():inspect(path+"/"+member,archive.read(member))
            return
        require(not seed_pattern.search(raw) and all(h not in raw for h in patterns[2:]),"Validation map found in new bytes: "+path)
        if "/replays.tar/replays/" in path:
            tape=json.loads(raw)
            values=tape["steps"][0][0]["observation"]["full_carbon"]
            digest=REG.map_identity.terrain_sha256(values)
            require(digest not in boards.values(),"Validation layout in archived original replay")
            replay_openings[Path(path).stem]=digest
    for name in paths:
        if not name:continue
        path=name.decode()
        require(all(h not in name for h in patterns),"Validation map named in changed path")
        raw=REG.git("show",SOURCE+":"+path)
        manifest[path]=hashlib.sha256(raw).hexdigest();inspect(path,raw)
    require(len(replay_openings)==2,"Expected the two newly archived complete original replays")
    return {"start":CENSUS,"end":SOURCE,"file_sha256":manifest,"new_raw_replay_layouts":replay_openings}

def finish(census,delta):
    require(census["source_snapshot"]==CENSUS and delta["start"]==PRIOR and delta["end"]==SOURCE,"Wrong snapshot chain")
    require(not census["seed_intersections"] and not census["physical_intersections"] and len(set(census["by_seed"].values()))==2,"Overlap in complete repository census")
    tested_seeds=set(SEEDS); tested_boards=set(census["by_seed"].values())
    groups={}
    def exclude(label,pairs):
        pairs=list(pairs)
        for seed,terrain in pairs:
            require(type(seed) is int and re.fullmatch("[0-9a-f]{64}",terrain),"Invalid map in "+label)
            require(seed not in tested_seeds and terrain not in tested_boards,"Previously exposed map in "+label)
        groups[label]={"records":len(pairs),"distinct_terrains":len({h for _,h in pairs})}
    intake,ip=bound(INTAKE/"protocol.json")
    past,ph=bound(INTAKE/"clearance.json")
    require(ip==REGISTRY_SHA and past["source_snapshot"]==PRIOR and intake["expected_games"]==24 and sorted(intake["seeds"])==list(range(621758460,621758466)),"Wrong active original training intake")
    require(intake["physical_boards"]==past["by_seed"]==past["permanently_excluded_from_independent_evaluation"]["physical_boards"],"Training and exclusion physical maps differ")
    require(len(intake["schedule"])==12 and {(s,p) for s in intake["seeds"] for p in (0,1)}=={(x["seed"],x["seat"]) for x in intake["schedule"]},"Missing active training ticket")
    exclude("24 original training games on six permanently retired boards",((int(s),h) for s,h in past["by_seed"].items()))
    league,lh=bound(LEAGUE/"clearance.json")
    exclude("18 league training, 6 league development and 2 original reserves",((int(s),h) for s,h in league["by_seed"].items()))
    repeated,rh=bound(REP.HERE/"clearance.json")
    exclude("fresh natural-policy development 450 and 451",((int(s),h) for s,h in repeated["by_seed"].items()))
    earlier,eh=bound(REP.HERE/"pre_dispatch_increment.json")
    def openings(rows):return ((r["seed"],r["opening_225_u8_sha256"]) for r in rows)
    exclude("previous inclusive ladder score-reference physical census",openings(repeated["latest_score_delta"]["verified_replays"]))
    exclude("up to intake registration ladder score changes",openings(past["new_score_delta"]["verified_replays"]))
    require(earlier["through_sha"]==prior.PREVIOUS and earlier["frozen_clearance_sha256"]==rh,"Prior publication advance mismatch")
    archived=[dict(game=r["id"],seat_names=r["physical_seats"],seed=r["seed"],opening_225_u8_sha256=r["layout_sha256"])
              for r in earlier["new_archived_ladder_replays"]]
    exclude("prior pre-dispatch archived official replay openings",openings(archived))
    recent=delta["verified_replays"]
    require({r["game"] for r in recent}==set(delta["compared_ids_including_mirrors"]) and len(recent)==len(delta["compared_ids_including_mirrors"]),"Incomplete new ladder opening check")
    exclude("new references through current main",openings(recent))
    reserve=census["formal_seed_half_open"]
    require(all(not reserve[0]<=s<reserve[1] for s in SEEDS),"Official exam seed overlap")
    originals,oh=bound(LEAGUE/"native_gradient/balanced_step/original_response/protocol.json")
    for group,boards in originals["training_physical_maps"].items():
        exclude("balanced candidate fitted maps "+group,((int(s),h) for s,h in boards.items()))
    exclude("balanced candidate original response diagnostics",((r["seed"],r["terrain"]) for r in originals["cases"]))
    base_games=set()
    base_terrains=set()
    for path in REG.git("ls-tree","-r","--name-only",SOURCE).decode().splitlines():
        if "/all_top5/" in path and path.endswith("/validation_maps.jsonl.gz"):
            data=gzip.decompress(REG.git("show",SOURCE+":"+path))
            for line in data.splitlines():
                item=json.loads(line);base_games.add(item["id"]);base_terrains.add(item["initial_terrain_u8_sha256"])
    previous_top=set(repeated["latest_score_delta"]["first_seen_top5"])|set(past["new_score_delta"]["first_seen_top5"])
    # There may be newly archived references after the previous census whose score already existed.
    for rows in (repeated["latest_score_delta"]["verified_replays"],past["new_score_delta"]["verified_replays"],archived,recent):
        for item in rows:
            names=item.get("seat_names",())
            if set(names)&set(REP.fetch.NAMES):
                base_games.add(item["game"]);base_terrains.add(item["opening_225_u8_sha256"])
    require(previous_top<=base_games and set(delta["first_seen_top5"])<=base_games and len(base_games)>=past["top5_total_distinct_game_ids_including_faults"],
            "Missing top-five ledger or previous score-delta identity")
    require(not(tested_boards&base_terrains),"Reused top-five terrain")
    baseline_faults=set(census["top5_faults"])
    count=len(base_games)
    changed=changed_evidence(census["by_seed"])
    require(set(changed["new_raw_replay_layouts"])<=base_games and set(changed["new_raw_replay_layouts"].values())<=base_terrains,
            "Raw replay openings do not match the complete registry")
    return {**census,"historical_census_snapshot":CENSUS,"historical_census_tree":census["source_tree"],
        "source_tree":REG.git("rev-parse",SOURCE+"^{tree}").decode().strip(),"new_changed_files_audit":changed,"seeds":SEEDS,"snapshot":SOURCE,"source_snapshot":SOURCE,
        "purpose":"First-use original-opponent development only for the future trained policy and its frozen control. Both seeds and opening layouts stay out of training and the independent final examination.",
        "by_seed":census["by_seed"],"physical_boards":census["by_seed"],"exclusion_groups":groups,
        "top5_games_scored":count-len(baseline_faults),"top5_total_distinct_game_ids_including_faults":count,"top5_unique_openings":len(base_terrains),
        "top5_new_scored_game_ids":delta["first_seen_top5"],"latest_score_delta":delta,
        "active_training":{"seeds":intake["seeds"],"physical_boards":intake["physical_boards"],"games":24,"tickets":12,"protocol_sha256":ip},
        "bound_prior_files":{"active_intake_clearance":ph,"league_clearance":lh,"natural_repeat_clearance":rh,
        "repeat_dispatch_increment":eh,"balanced_response_protocol":oh},
        "coverage":"Repository-visible registered or recorded maps plus linked public score additions through the frozen source commit. Unpublished private games are outside this inventory."}

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--scratch",type=Path,required=True);cli.add_argument("--stage",choices=("census","recent","finish","all"),default="all")
    args=cli.parse_args();work=args.scratch;work.mkdir(parents=True,exist_ok=True)
    if args.stage in ("census","all"):REP.freeze(work/"census.json",census())
    if args.stage in ("recent","all"):REP.freeze(work/"recent.json",score_delta(work))
    if args.stage in ("finish","all"):
        REP.freeze(HERE/"clearance.json",finish(json.loads((work/"census.json").read_bytes()),json.loads((work/"recent.json").read_bytes())))
if __name__=="__main__":main()
