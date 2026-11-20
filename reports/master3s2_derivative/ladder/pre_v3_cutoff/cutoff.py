"""Read-only two-ID official-scorecard census and one-time new-loss reconciliation."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import gzip, hashlib, json, sys

HERE = Path(__file__).resolve().parent
LADDER = HERE.parent
sys.path.insert(0, str(LADDER/"placement_refresh"))
import refresh
from recover_public import extract,canonical

IDS = {"v1":"b99742a4a510a4de5dc99639b3c85604","v2":"ec94599e5182be7d35eb14bc288625cd"}
HISTORY = {"v1": LADDER/"placement_refresh/summary.json", "v2":LADDER/"v2_loss_review/summary.json"}
def stamp():
    return datetime.now(timezone.utc).isoformat()
def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(name):
    pages=[]
    offset=0
    while True:
        url=f"{refresh.BASE}/api/matches?limit=20&offset={offset}&strategy={IDS[name]}"
        data=json.loads(refresh.get(url))
        pages.append(dict(url=url,read_utc=stamp(),data=data))
        if data["next_offset"] is None:
            break
        assert data["next_offset"]>offset
        offset=data["next_offset"]
    rows=[m for p in pages for m in p["data"]["matches"]]
    assert len(rows)==len({m["id"] for m in rows})
    refresh.store(HERE/f"{name}_pages.json.gz",pages)


def fetch_all():
    cutoff=stamp()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch,IDS))
    refresh.store(HERE/"cutoff.json",dict(shared_utc=cutoff,returned_utc=stamp(),ids=IDS))


def reconcile(name,cutoff,online):
    send_id=IDS[name]
    refresh.ID=send_id
    path=HERE/f"{name}_pages.json.gz"
    pages=refresh.read(path);raws=[m for p in pages for m in p["data"]["matches"]]
    limit=datetime.fromisoformat(cutoff).timestamp()
    hist=refresh.read(HISTORY[name])
    old={(c["match"],c["game"]):c for c in hist["all_settled_games" if name=="v1" else "games"]}
    oldloss={key for key,c in old.items() if c["outcome"]=="L"}
    rows=[];pending=[];fresh=[];settled=set()
    for m in sorted(raws,key=lambda z:z["created"]):
        if m["created"]>limit:
            continue
        assert send_id in (m["a"],m["b"])
        if m["state"]!="done" or not m["result"] or m["finished"]>limit:
            pending.append({k:m[k] for k in ("id","kind","a","b","name_a","name_b","created")})
            continue
        assert len(m["result"])==m["game_count"]
        for g in m["result"]:
            c=refresh.score(m,g)
            key=(c["match"],c["game"])
            assert key not in settled
            settled.add(key)
            rows.append(c)
            if key in old:
                prior=old[key]
                assert all(prior[k]==c[k] for k in ("seat","outcome","score","margin","opponent_id")),key
            if c["outcome"]!="L" or key in oldloss:
                continue
            fn=f"{key[0]}-{key[1]}.json.gz"
            archive=HERE/"replays"/fn
            if not archive.exists():
                for cached in [LADDER/"replays"/fn,LADDER/"v2_loss_review/replays"/fn,LADDER/"placement_refresh/replays"/fn]:
                    if cached.exists():
                        archive=cached
                        break
                else:
                    if not online:
                        raise ValueError(f"missing public replay {fn}")
                    replay,_,_=extract(refresh.get(f"{refresh.BASE}/replay/{key[0]}/{key[1]}"))
                    refresh.validate(replay,{**m,**g})
                    archive.parent.mkdir(exist_ok=True)
                    archive.write_bytes(gzip.compress(canonical(replay),compresslevel=6,mtime=0))
            replay=refresh.read(archive)
            check=refresh.validate(replay,{**m,**g})
            c["replay_sha256"]=digest(archive.read_bytes())
            c["replay_path"]=str(archive.relative_to(LADDER))
            c["verified"]=check
            c["opening_terrain_sha256"]=digest(bytes(replay["steps"][0][0]["observation"]["full_carbon"]))
            if c["steps"]==299 and set(c["statuses"].values())=={"DONE"}:
                refresh.compare.HERE=archive.parent.parent/"analysis"
                detail=refresh.review.losing_game(dict(opponent=c["opponent"],match_id=c["match"],seat=c["seat"],steps=c["steps"]),archive)
            else:
                proof=refresh.review.validation.official({"path":archive.name},archive.parent)
                assert proof["matched"]==c["steps"] and proof["first_divergence"] is None
                detail={"official":proof,"steps_verified":c["steps"],"archive":"see both statuses; unobserved counterfactuals have no final score"}
            detail["card"]=c
            out=HERE/(name+"_"+key[0]+f"-{key[1]}.ledger.json.gz")
            refresh.store(out,detail)
            c["ledger_path"]=out.name
            fresh.append(c)
    assert set(old)<=settled
    groups={}
    for kind in sorted({c["kind"] for c in rows+pending}):
        group=[c for c in rows if c["kind"]==kind]
        groups[kind]=dict(done=len(group),outcomes=dict(Counter(c["outcome"] for c in group)),pending=sum(p["kind"]==kind for p in pending),
                           seats={s:dict(Counter(c["outcome"] for c in group if c["seat"]==s)) for s in (0,1)})
    return dict(id=send_id,raw_pages_sha256=digest(path.read_bytes()),old_summary_sha256=digest(HISTORY[name].read_bytes()),
        first_response=pages[0]["read_utc"],last_response=pages[-1]["read_utc"],groups=groups,
        old_completed=len(old),old_losses=len(oldloss),new_losses=fresh,pending=pending,all_games=rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fetch",action="store_true")
    parser.add_argument("--download",action="store_true")
    opts=parser.parse_args()
    if opts.fetch:
        fetch_all()
    cut=refresh.read(HERE/"cutoff.json");results={name:reconcile(name,cut["shared_utc"],opts.download) for name in IDS}
    refresh.store(HERE/"summary.json",dict(**cut,strategies=results))
    for name,r in results.items():
        print(name,r["groups"],[(c["opponent"], c["match"],c["margin"]) for c in r["new_losses"]])
if __name__=="__main__":
    main()
