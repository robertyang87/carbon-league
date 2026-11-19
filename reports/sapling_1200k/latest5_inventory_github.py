"""Download verified replay data for frozen GitHub/leaderboard card deltas."""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path
import ssl
import subprocess
import sys
import tempfile
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"reports/master_family/deep_review"))
from recover_public import extract, validate, canonical, key_of, walk_matches, identity

CACHE=Path("/workspace/sapling-latest5-nov19")
REPORT=Path(__file__).resolve().parent
DATA=ROOT/"data/sapling_latest5_20261119"
BASE="a0124f50"
HEAD="ce394625"
SERVER="https://117.72.210.134"
INDEXES=("master5s","master6","it720","it720preview","m9xj630","fablev7")


def git(ref,path):
    return subprocess.check_output(["git","-C",str(ROOT),"show",ref+":"+path])


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def atomic(path,blob):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
        f.write(blob)
    Path(f.name).replace(path)


def dump(path,obj):
    atomic(path,(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode())


def rows(ref,names):
    records={}
    for n in INDEXES:
        for r in walk_matches(json.loads(git(ref,f"data/{n}_matches.json"))):
            if r["state"]!="done" or not ({r["name_a"],r["name_b"]}&names):
                continue
            k=key_of(r)
            if k in records and identity(records[k])!=identity(r):
                raise ValueError("conflicting duplicate score card "+k)
            records[k]=r
    return records


def references():
    frozen=json.loads(git(HEAD,"reports/master_family/tactical_rules/top5_inventory.json"))["ranks"]
    names={r["name"] for r in frozen}
    a,b=rows(BASE,names),rows(HEAD,names)
    old=json.loads(git(HEAD,"data/sapling_top5_h2h_corpus/sources.json"))["games"]
    prior={r["id"] for r in old}
    prior.update(row["game"] for row in json.loads(git(HEAD,"reports/sapling_1200k/ladder_cash_rules.json"))["live"])
    if len(prior)!=110:
        raise ValueError("prior history changed")
    return names,a,b,prior


def read_url(url):
    req=urllib.request.Request(url,headers={"Accept-Encoding":"gzip","User-Agent":"CarbonLeagueReplayReview/1.0"})
    with urllib.request.urlopen(req,context=ssl._create_unverified_context(),timeout=30) as f:
        return f.read(),f.headers.get("date")


def verified(item):
    k,card,cohort=item
    url=SERVER+f"/replay/{card['id']}/{card['game']}"
    cache=CACHE/"replays"/(k+".json.gz")
    if cache.exists():
        blob=cache.read_bytes();env=json.loads(gzip.decompress(blob))
    else:
        buf,date=read_url(url)
        env,_,_=extract(buf)
        blob=gzip.compress(canonical(env),compresslevel=6,mtime=0)
    detail=validate(env,card)
    if detail["statuses"]!=["DONE","DONE"] or not isinstance(detail["random_seed"],int):
        raise ValueError("execution fault or unusable map seed "+k)
    for i,pair in enumerate(env["steps"]):
        a,b=[seat["observation"] for seat in pair]
        if len(a["players"])!=2 or "full_carbon" not in a or "full_carbon" in b or "carbon" not in b or a.get("player")!=0 or b.get("player")!=1:
            raise ValueError(f"invalid observation visibility on {k}/{i}")
        if i and any(not isinstance(s["action"],dict) for s in pair):
            raise ValueError("missing joint commands "+k)
    atomic(cache,blob)
    return dict(id=k,seed=detail["random_seed"],seat_names=detail["seat_model_names"],path="replays/"+cache.name,
                sha256=sha(blob),canonical_sha256=sha(canonical(env)),cohort=cohort,eligible=True,
                environment_uuid=detail["environment_uuid"],transitions=detail["last_step"],
                terminal=detail["statuses"],score_by_seat=detail["rewards_by_seat"],
                visible_fields_by_seat=[sorted(env["steps"][-1][p]["observation"]) for p in (0,1)],
                fetched_from=url,created=card["created"],card=card)


def batch(items):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        return sorted(pool.map(verified,items),key=lambda r:r["id"])


def main():
    names,a,b,prior=references()
    cohort=lambda r:"github_top5_mutual" if {r["name_a"],r["name_b"]}<=names else "github_top5_external"
    new=set(b)-set(a)
    games=batch([(k,b[k],cohort(b[k])) for k in sorted(new-prior)])
    out=dict(github_after=HEAD,github_before=BASE,index_additions=len(new),already_exposed=sorted(new&prior),
             games=games,scope="Incremental GitHub cards only. Local replay paths relative to the shared cache. Final manifest carries seed/UUID and previous-data collision checks.")
    dump(REPORT/"latest5_inventory_github.json",out)
    print([(x["id"],x["seed"]) for x in games],flush=True)


if __name__=="__main__":
    main()
