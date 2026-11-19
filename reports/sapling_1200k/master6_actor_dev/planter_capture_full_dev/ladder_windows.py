"""Link every ownership change in four signed replays to pre-action histories and eventual receipts."""
import gzip
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/"planter_payback_full_dev"
sys.path.insert(0,str(HERE.parents[1]/"all_top5"))
import fetch

def distance(a,b):
    dy,dx=abs(a//15-b//15),abs(a%15-b%15)
    return min(dx,15-dx)+min(dy,15-dy)

def one(r):
    key=r["id"]
    cache=fetch.CACHE/"replays"/(key+".json.gz")
    raw=cache.read_bytes() if cache.exists() else fetch.Retrieval({},0).get(r["url"])
    env,_,_=fetch.extract(raw)
    canonical=fetch.canonical(env)
    assert fetch.sha(canonical)==r["canonical_sha256"]
    if not cache.exists():
        tmp=cache.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(canonical,mtime=0))
        tmp.replace(cache)
    steps=env["steps"]
    cases=[]
    for tid,tree in r["trees"].items():
        site=tree["site"]
        for sp in tree["spans"][1:]:
            s=sp["seat"]
            root=sp["start"]-1
            wid=steps[root+1][0]["observation"]["trees"][tid][0]
            assert wid is not None
            history=[]
            for t in range(max(0,root-10),root+1):
                ob=steps[t][0]["observation"]
                own=ob["players"][s]
                enemy=ob["players"][1-s]
                worker=own[2].get(wid)
                threat=sorted((distance(w[0],site),uid,w[0],w[2]) for uid,w in enemy[2].items())
                history.append(dict(step=t,worker=worker,command=(steps[t+1][s].get("action") or {}).get(wid),
                                    nearest_enemies=threat[:3],cash=[own[0],enemy[0]],
                                    previous_payment=ob["trees"].get(tid,[None,None])[1]))
            assert history[-1]["worker"][0]==site and history[-1]["command"] in (None,"NONE")
            cases.append(dict(tree=tid,site=site,seat=s,step=root,worker=wid,history=history,
                              holding_turns=sp["end"]-sp["start"]+1,realized_gross=sp["income"]))
    realized={(x["step"],x["seat"],x["worker"],x["tree"]):x for x in cases}
    opportunities=[]
    for t in range(len(steps)-1):
        players=steps[t][0]["observation"]["players"]
        for seat,own in enumerate(players):
            if own[0]<20:
                continue
            enemy_trees={pos:(tid,age) for tid,(pos,age) in players[1-seat][3].items() if age<50}
            for wid,(pos,load,kind) in own[2].items():
                if kind!="PLANTER" or pos not in enemy_trees:
                    continue
                tid,age=enemy_trees[pos]
                command=(steps[t+1][seat].get("action") or {}).get(wid)
                event=realized.get((t,seat,wid,tid))
                opportunities.append(dict(step=t,seat=seat,worker=wid,tree=tid,site=pos,age=age,
                                          command=command,transferred=bool(event),
                                          gross=event["realized_gross"] if event else None))
    return dict(id=key,replay_sha256=r["canonical_sha256"],cases=cases,opportunities=opportunities)

if __name__=="__main__":
    rows=json.loads(gzip.decompress((PREVIOUS/"ladder_trees.json.gz").read_bytes()))
    with ThreadPoolExecutor(2) as pool:
        out=list(pool.map(one,rows))
    (HERE/"ladder_windows.json.gz").write_bytes(gzip.compress(fetch.canonical(out),mtime=0))
    print([len(r["cases"]) for r in out])
