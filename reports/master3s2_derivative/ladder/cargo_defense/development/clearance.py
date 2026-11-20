"""Screen the two development openings against known terrain, registrations and both live ladder entrants."""
from datetime import datetime, timezone
import gzip
import importlib.util
import json
from pathlib import Path
import re
import ssl
import urllib.request

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parents[2]
LADDER=FAMILY/"ladder"
SOURCE="91d1e13719060a50efd36cfc443510f0456c2b0c"
SEEDS=[621758594,621758595]
TOKEN=re.compile(rb"(?<![.\da-zA-Z])62175859[45](?![.\da-zA-Z])")
SERVER="https://117.72.210.134"

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

old=module("previous_full_archive_census",FAMILY/"cold_start/development/clearance.py")
prior=old.prior
public=module("official_replay",FAMILY.parent/"master_family/deep_review/recover_public.py")

def require(test,reason):
    if not test:raise ValueError(reason)

def full_census():
    old.BASE,old.SEEDS,old.TOKEN=SOURCE,SEEDS,TOKEN
    return old.generate()

def get(url):
    request=urllib.request.Request(url,headers={"User-Agent":"carbon-league/board-census"})
    with urllib.request.urlopen(request,context=ssl._create_unverified_context(),timeout=25) as response:
        return response.read()

def live_cards(uid):
    at=0
    offsets=set()
    rows={}
    while True:
        require(at not in offsets,"Repeated leaderboard page")
        offsets.add(at)
        page=json.loads(get(f"{SERVER}/api/matches?strategy={uid}&limit=100&offset={at}"))
        for row in page["matches"]:
            require(uid in (row["a"],row["b"]),"Incorrect filtered player")
            rows[row["id"]]=row
        at=page["next_offset"]
        if at is None:break
    return rows

def complete(census,endpoint):
    boards=census["by_seed"]
    require(census["source_snapshot"]==SOURCE and set(map(int,boards))==set(SEEDS),"Wrong initial census")
    require(not census["seed_intersections"] and not census["physical_intersections"],"Historical overlap")
    prior.SEEDS=prior.old.SEEDS=SEEDS
    prior.IDS=TOKEN
    delta=dict(by_seed=boards,remote_extension=[])
    prior.extend(delta,SOURCE,endpoint)
    added=prior.verify_added_boards(SOURCE,endpoint,boards)
    def frozen(p):return prior.frozen(p,endpoint)
    def admit(seed_ids,terrains):
        require(not set(seed_ids)&set(SEEDS) and not set(terrains)&set(boards.values()),"Seed or opening collision")
    previous={}
    for path in (FAMILY/"dev1/protocol.json",FAMILY/"cold_start/development/protocol.json",
                 LADDER/"prediction_feedback/development/protocol.json"):
        raw=frozen(path);reg=json.loads(raw);known=reg["physical_boards"]
        admit(reg["seeds"],known.values())
        previous[path.relative_to(prior.ROOT).as_posix()]={"sha256":prior.sha(raw),"maps":known}
    cards=json.loads(frozen(LADDER/"ds-pro-v1-placement.json"))
    archive={}
    known_names=set(prior.git("ls-tree","-r","--name-only",endpoint).decode().splitlines())
    def record(match,game,score_seed,score_map,source_card=None):
        key=f"{match}-{game}"
        require(re.fullmatch(r"[a-f0-9]{32}-[1-9][0-9]*",key),"Invalid replay key")
        rel=LADDER/f"replays/{key}.json.gz"
        path=rel.relative_to(prior.ROOT).as_posix()
        url=SERVER+f"/replay/{match}/{game}"
        if key in archive:
            prev=archive[key]
            require((score_seed,score_map)==(prev["platform_map_seed"],prev["platform_map_sha256"]),"Conflicting live scorecard")
            return
        raw=frozen(rel) if path in known_names else get(url)
        env,_,_=public.extract(raw)
        if source_card:public.validate(env,source_card)
        initial=env["steps"][0][0]["observation"]
        require(initial.get("step")==0,"Missing opening")
        terrain=prior.old.engine.map_identity.terrain_sha256(initial["full_carbon"])
        replay_seed=env["configuration"]["randomSeed"]
        admit((score_seed,replay_seed),(terrain,score_map))
        archive[key]=dict(match=match,game=game,platform_map_seed=score_seed,platform_map_sha256=score_map,
                          replay_random_seed=replay_seed,opening_u8_sha256=terrain,source_sha256=prior.sha(raw),
                          source=path if path in known_names else url)
    for r in cards["games"]:
        record(r["match_id"],r["game"],r["map_seed"],r["map_sha256"])
    require(len(archive)==cards["settled_games"],"Incomplete repository placement history")
    live={}
    for version in ("ds-pro-v1","ds-pro-v2"):
        uid=json.loads(frozen(LADDER/(version+".json")))["platform"]["id"]
        rows=live_cards(uid);complete_keys=[];pending={}
        for match,row in rows.items():
            results=row.get("result") or []
            for g in results:
                record(match,g["game"],g["seed"],g["map_sha256"],{**row,**g})
                complete_keys.append(f"{match}-{g['game']}")
            if row["state"]!="done" or not results:pending[match]=row["state"]
        live[version]=dict(strategy_id=uid,endpoint=SERVER+"/api/matches?strategy="+uid,
                           completed=sorted(complete_keys),pending_without_complete_result=pending)
    fit={}
    plan=json.loads(frozen(HERE.parent/"protocol.json"))
    for split,rows in plan["maps"].items():
        fit[split]=[]
        for row in rows:
            key=Path(row["source"]).name.removesuffix(".json.gz")
            r=archive.get(key)
            if r is None:
                raw=frozen(LADDER/row["source"])
                require(prior.sha(raw)==row["sha256"],"Different non-ladder training game")
                env,_,_=public.extract(raw)
                terra=prior.old.engine.map_identity.terrain_sha256(env["steps"][0][0]["observation"]["full_carbon"])
                admit([env["configuration"]["randomSeed"]],[terra])
            else:
                require(r["source_sha256"]==row["sha256"],"Fit source does not equal the archived physical game")
                terra=r["opening_u8_sha256"]
            fit[split].append(dict(source=row["source"],source_sha256=row["sha256"],terrain=terra))
    census.update(schema="master3s2-cargo-defense/development-clearance/1",through=endpoint,boards=boards,
        purpose="These two maps are reserved only for development and excluded from training and the independent final exam.",
        groups={"cargo_defense_development_only":SEEDS},verifier_sha256=prior.sha(Path(__file__).read_bytes()),
        full_census_helper_sha256=prior.sha((FAMILY/"cold_start/development/clearance.py").read_bytes()),
        previous_development=previous,added_repo_replays=added,repo_extension=delta["remote_extension"],
        ladder_as_of_utc=datetime.now(timezone.utc).isoformat(),ladder_scorecards=live,ladder_openings=archive,
        cargo_defense_source_maps=fit,passed=True)
    return census

if __name__=="__main__":
    result=complete(full_census(),prior.checked_remote())
    print(old.shared.save(HERE/"clearance.json",result),result["boards"])
