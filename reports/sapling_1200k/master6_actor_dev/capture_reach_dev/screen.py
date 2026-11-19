"""Check a two-step reach screen on all actual planter approaches in four public matches."""
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"all_top5"))
import fetch

def dist(a,b):
    y,x = abs(a//15-b//15),abs(a%15-b%15)
    return min(y,15-y)+min(x,15-x)

def destination(pos,cmd):
    y,x = divmod(pos,15)
    dy,dx = {"UP":(-1,0),"DOWN":(1,0),"LEFT":(0,-1),"RIGHT":(0,1)}[cmd]
    return ((y+dy)%15)*15+(x+dx)%15

def inspect_game(doc):
    env,_,_ = fetch.extract((fetch.CACHE/"replays"/(doc["id"]+".json.gz")).read_bytes())
    assert fetch.sha(fetch.canonical(env)) == doc["canonical_sha256"]
    steps=env["steps"]
    rows=[]
    for t in range(len(steps)-2):
        players=steps[t][0]["observation"]["players"]
        for s,own in enumerate(players):
            en=players[1-s]
            enemy_sites={v[0]:key for key,v in en[3].items() if v[1]<49}
            for worker,(pos,cargo,kind) in own[2].items():
                cmd=(steps[t+1][s].get("action") or {}).get(worker)
                if kind!="PLANTER" or cmd not in ("LEFT","RIGHT","UP","DOWN") or own[0]<20:
                    continue
                site=destination(pos,cmd)
                if site not in enemy_sites:
                    continue
                nearest=min((dist(site,w[0]) for w in en[2].values()),default=99)
                if nearest<2:
                    continue
                tid=enemy_sites[site]
                span=next((sp for sp in doc["trees"][tid]["spans"] if sp["start"]==t+2 and sp["seat"]==s),None)
                owner=steps[t+2][0]["observation"]["trees"].get(tid,[None])[0]
                realized=span is not None and owner==worker
                rows.append(dict(step=t,seat=s,worker=worker,tree=tid,nearest=nearest,
                                 captured=realized,gross=span["income"] if realized else 0.))
    return dict(id=doc["id"],attempts=rows)

if __name__=="__main__":
    docs=json.loads(gzip.decompress((HERE.parent/"planter_payback_full_dev/ladder_trees.json.gz").read_bytes()))
    out=[inspect_game(d) for d in docs]
    (HERE/"screen.json.gz").write_bytes(gzip.compress(fetch.canonical(out),mtime=0))
    for r in out:
        for block in (True,False):
            a=[v for v in r["attempts"] if (v["nearest"]==2)==block]
            print(r["id"][:6],block,len(a),sum(v["captured"] for v in a),round(sum(v["gross"] for v in a)))
