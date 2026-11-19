"""Repeat the approach test for Master6 on every archived verified, scored appearance."""
from screen import HERE,fetch,dist,destination
import gzip
import json
from collections import Counter

def one(row,seat,data):
    env,_,_=fetch.extract(data)
    steps=env["steps"]
    counts=Counter()
    for t in range(len(steps)-2):
        players=steps[t][0]["observation"]["players"]
        own,enemy=players[seat],players[1-seat]
        if own[0]<20:continue
        sites={v[0]:k for k,v in enemy[3].items() if v[1]<49}
        for wid,(pos,_,kind) in own[2].items():
            if kind!="PLANTER":continue
            cmd=(steps[t+1][seat].get("action") or {}).get(wid)
            if cmd not in ("LEFT","RIGHT","UP","DOWN"):continue
            site=destination(pos,cmd)
            if site not in sites:continue
            nearest=min((dist(site,w[0]) for w in enemy[2].values()),default=99)
            if nearest<2:continue
            group="two" if nearest==2 else "far"
            counts[group+"_approach"]+=1
            tid=sites[site]
            post=steps[t+2][0]["observation"]
            if tid in post["players"][seat][3] and post["trees"][tid][0]==wid:
                counts[group+"_capture"]+=1
    return dict(id=row["id"],seat=seat,score=row["score_by_seat"],opponent=row["seat_names"][1-seat],seed=row["seed"],counts=dict(counts))

if __name__=="__main__":
    root=HERE.parents[1]/"all_top5"
    rows={}
    for p in [root/"manifest.json",*sorted(root.glob("appendix_*/manifest.json"))]:
        for row in json.loads(p.read_bytes())["games"]:
            if row.get("frame_verified") and "claude-AlphaCarbon-Master6" in row["seat_names"]:
                rows[row["id"]]=row
    result=[];missing=[]
    for k,r in sorted(rows.items()):
        p=fetch.CACHE/"replays"/(k+".json.gz")
        if not p.is_file():
            missing.append(k);continue
        data=p.read_bytes()
        if fetch.sha(data)!=r["sha256"]:
            raise ValueError(k+" archived SHA mismatch")
        for seat,name in enumerate(r["seat_names"]):
            if name=="claude-AlphaCarbon-Master6":
                result.append(one(r,seat,data))
    doc=dict(eligible_ids=len(rows),unavailable=missing,appearances=result)
    (HERE/"master6.json.gz").write_bytes(gzip.compress(fetch.canonical(doc),mtime=0))
    print(len(rows),len(missing),len(result))
