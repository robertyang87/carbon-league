"""Verify two development openings against a frozen remote tree, public replay updates and archived map registries."""
import argparse
import gzip
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import types
import zipfile

HERE = Path(__file__).resolve().parent
STRATEGY = HERE.parents[1]
BASE = STRATEGY/"policy_window_returns"
ELITE = STRATEGY/"elite_direct_policy/development/clearance.py"
TEMPLATE = STRATEGY.parent/"cargo_tree_action_learning_dev/native_actor_corpus/league_policy/clearance.py"
ROOT = next(p for p in HERE.parents if (p/".git").exists())
SEEDS = [621758550, 621758551]
FROZEN = "2f88952886a9a309f06b7ec5b8cfc5ad633b0ac4"
IDS = re.compile(rb"(?<![.\da-zA-Z])62175855[01](?![.\da-zA-Z])")


def register(path):
    spec = importlib.util.spec_from_file_location("signed_map_census", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


old = register(ELITE)
old.SEEDS = SEEDS
git = old.git
sha = old.sha


def checked_remote():
    out = git("ls-remote", "origin", "refs/heads/main").decode().split()
    if len(out) != 2 or out[1] != "refs/heads/main":
        raise RuntimeError("remote main identity could not be read")
    subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", out[0]+"^{commit}"], check=True)
    return out[0]


def frozen(path, commit):
    return git("show", commit+":"+str(path.relative_to(ROOT)))


def admit(rows, roots):
    for row in rows:
        if row["seed"] in SEEDS or row["terrain"] in roots.values():
            raise ValueError(f'Existing physical map or seed in {row}')


def crawl(snapshot):
    old.SOURCE = snapshot
    names = git("ls-tree","-r","--name-only",snapshot).decode().splitlines()
    canonical, proofs, faults = old.top_ledgers(names)
    raw = frozen(TEMPLATE,snapshot)
    program = raw.decode()
    changes = {
        "len(top5)==1637": f"len(top5)=={len(canonical)}",
        "if b'6217584' in b and CANDIDATE.search(b)": "if b'62175855' in b and CANDIDATE.search(b)",
    }
    for pattern,replacement in changes.items():
        if program.count(pattern)!=1:
            raise RuntimeError("Upstream archive scanner has changed")
        program=program.replace(pattern,replacement)
    lib=types.ModuleType("matched_economic_map_census")
    lib.__file__=str(TEMPLATE)
    exec(compile(program,str(TEMPLATE),"exec"),lib.__dict__)
    lib.SNAPSHOT,lib.SEEDS,lib.GROUPS,lib.CANDIDATE=snapshot,SEEDS,{"original_development":SEEDS},IDS
    census=lib.scan()
    roots=old.boards()
    if census["by_seed"]!=roots or len(canonical)<1649:
        raise ValueError("Incorrect opening denominator or elementary board reproduction")
    census.update(schema="sapling-midgame-economic-two-development-openings/1",purpose="Two fresh development maps only. Exclude from independent final examination and from training.",canonical_ledgers=proofs,canonical_faults=faults,census_template_sha256=sha(raw))
    prior_path=BASE/"clearance_increment.json"
    prior=json.loads(frozen(prior_path,snapshot))
    capture=json.loads(frozen(BASE/"clearance.json",snapshot))
    if list(sorted(map(int,capture["by_seed"]))) != list(range(621758540,621758548)) or prior["physical_boards"]!=capture["by_seed"]:
        raise ValueError("New eight-map panel incomplete")
    admit([dict(seed=int(k),terrain=v) for k,v in capture["by_seed"].items()],roots)
    receipts=[(STRATEGY/"elite_direct_policy/development/score_delta.json",()),
              (BASE/"clearance.json",("score_increment",)),(prior_path,("score_increment",))]
    known={}
    for path,keys in receipts:
        data=frozen(path,snapshot);p=json.loads(data)
        for k in keys:
            p=p[k]
        if p["unresolved_incremental_replays"]:
            raise ValueError("Historical score clearance was incomplete")
        rows=p["additional_public_replays"]
        admit(rows,roots)
        known[str(path.relative_to(ROOT))]=dict(sha256=sha(data),before=p["base"],through=p["through"],rows=len(rows))
    old.BASE=prior["through"]
    score=old.score_delta()
    census.update(added_training_and_holdout_maps=capture["by_seed"],prior_increment_receipts=known,
                  official_score_delta=score,remote_extension=[])
    return census


def extend(census, start, through):
    if start==through:
        return
    old.BASE,old.SOURCE=start,through
    git("merge-base","--is-ancestor",start,through)
    paths=git("ls-tree","-r","--name-only",through).decode().splitlines()
    identities,registry,faults=old.top_ledgers(paths)
    roots=census["by_seed"]
    admit((dict(seed=seed,terrain=h) for seed,h in identities.values()),roots)
    score=old.score_delta()
    changed=git("diff","--diff-filter=ACMRT","--name-only",start,through).decode().splitlines()
    hashes=[v.encode() for v in roots.values()]
    def inspect(path,data):
        if IDS.search(path.encode()) or IDS.search(data) or any(h in data for h in hashes):
            raise ValueError("New source contains registered identity: "+path)
        if path.endswith((".tgz",".tar.gz",".tar")):
            with tarfile.open(fileobj=io.BytesIO(data)) as tar:
                for m in tar:
                    if m.isfile():
                        inspect(m.name,tar.extractfile(m).read())
        elif path.endswith((".zip",".npz")):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for name in z.namelist():
                    inspect(name,z.read(name))
        elif path.endswith(".gz"):
            inspect(path[:-3],gzip.decompress(data))
    listed={}
    reservations={}
    for path in changed:
        raw=git("show",through+":"+path)
        h=sha(raw)
        if path==str((HERE/"design.json").relative_to(ROOT)) and h=="952cf49aac8e8a934596ca9bd01cf73ffe1a473caa91225e4b6a6fabf99f4a28":
            # This specific registered schedule was reviewed before any score existed.
            design=json.loads(raw)
            if design["physical_boards"]!=roots or design["seeds"]!=SEEDS or len(design["schedule"])!=4:
                raise ValueError("Different preregistration")
            reservations[path]=h
        else:
            inspect(path,raw)
        listed[path]=h
    census["remote_extension"].append(dict(after=start,through=through,changed_sha256=listed,top5_ledgers=registry,
        top5_unique_openings=len(identities),top5_faults=faults,score_delta=score,own_exact_preregistration=reservations))


def verify_added_boards(source, endpoint, roots):
    """Hash physical openings even for gap-fill replays that add no new scorecard ID."""
    changed = git("diff","--name-only","--diff-filter=ACMRT",source,endpoint).decode().splitlines()
    layouts = {}
    def visit(name, data):
        if name.endswith((".tgz",".tar.gz",".tar")):
            with tarfile.open(fileobj=io.BytesIO(data)) as z:
                for item in z:
                    if item.isfile():
                        visit(name+"!"+item.name,z.extractfile(item).read())
        elif name.endswith((".zip",".npz")):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for item in z.namelist():
                    visit(name+"!"+item,z.read(item))
        elif name.endswith(".gz"):
            visit(name[:-3],gzip.decompress(data))
        elif name.endswith(".json") and b'"full_carbon"' in data and b'"steps"' in data:
            game=json.loads(data)
            if not isinstance(game,dict) or not isinstance(game.get("steps"),list) or not game["steps"]:
                return
            step=game["steps"][0]
            if not isinstance(step,list) or not step or not isinstance(step[0],dict):
                return
            opening=step[0].get("observation",{})
            cells=opening.get("full_carbon")
            if cells is not None and opening.get("step",0)==0:
                h=old.engine.map_identity.terrain_sha256(cells)
                if h in roots.values():
                    raise ValueError("New replay physical terrain collision in "+name)
                layouts[name]=dict(game_uuid=game.get("id"),seed=game.get("configuration",{}).get("randomSeed"),terrain=h)
    for path in changed:
        if path.endswith((".json",".json.gz",".tgz",".tar.gz",".tar",".zip",".npz")):
            visit(path,git("show",endpoint+":"+path))
    return layouts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through",help="Reproduce the published endpoint (default reads remote main before and after materialization).")
    parser.add_argument("--verified-baseline", type=Path, help="Resume a completed full census from this file (baseline SHA must be supplied).")
    parser.add_argument("--baseline-sha", help="Expected byte SHA-256 of the completed census.")
    args=parser.parse_args()
    observed=args.through or checked_remote()
    git("merge-base","--is-ancestor",FROZEN,observed)
    dest=HERE/"clearance.json"
    if args.verified_baseline:
        raw=args.verified_baseline.read_bytes()
        if sha(raw)!=args.baseline_sha:
            raise ValueError("Wrong baseline receipt")
        out=json.loads(raw)
        if out["source_snapshot"]!=FROZEN or out["by_seed"]!=old.boards() or out["top5_unique_openings"]<1649:
            raise ValueError("Unrelated baseline or openings")
        start=out["through"]
    else:
        out=crawl(FROZEN)
        start=FROZEN
    extend(out,start,observed)
    endpoint=args.through or checked_remote()
    while observed!=endpoint:
        extend(out,observed,endpoint)
        observed=endpoint
        endpoint=checked_remote()
    out["additional_raw_replay_layouts"]=verify_added_boards(FROZEN,endpoint,out["by_seed"])
    out.update(through=endpoint,through_tree=git("rev-parse",endpoint+"^{tree}").decode().strip(),code_sha256=sha(Path(__file__).read_bytes()),
               upstream_window_protocol_sha256=sha(frozen(BASE/"protocol.json",endpoint)),
               score_evidence_scope="Archived physical openings, recorded seed/hash declarations and identifiable new score references; public score cards alone need replay retrieval to establish the terrain.")
    dest=HERE/"clearance.json"
    raw=(json.dumps(out,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n").encode()
    if dest.exists() and dest.read_bytes()!=raw and not (args.verified_baseline and sha(dest.read_bytes())==args.baseline_sha):
        raise ValueError("Different receipt already exists")
    temp=dest.with_suffix(".partial")
    temp.write_bytes(raw); temp.replace(dest)
    print(out["through"],out["by_seed"],out["top5_unique_openings"],sha(raw),flush=True)


if __name__=="__main__":
    main()
