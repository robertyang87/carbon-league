"""Map- and opponent-separated splits, plus a small predeclared exact-Sapling test on ladder boards."""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parent
BASE=ACTOR.parent
TOP=BASE/"all_top5"
A=ACTOR/"tree_tenure_dev/pay_policy"
REPLAYS=Path("/workspace/sapling-all-top5-20261119/replays")
def read(p):
    b=p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix==".gz" else b)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(*items):
    return hashlib.sha256(json.dumps(items,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def save(p,obj):
    if p.exists():
        if read(p)!=obj:raise ValueError(f"immutable {p} differs")
        return
    data=(json.dumps(obj,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n").encode()
    if p.suffix==".gz":data=gzip.compress(data,mtime=0)
    tmp=p.with_suffix(p.suffix+".tmp");tmp.write_bytes(data);tmp.replace(p)
def main():
    prior=read(A/"opportunities.json.gz")
    blocked={r["terrain"] for r in prior["games"]}
    banned_seeds=set(read(A/"protocol.json")["new_dev"]["seeds"])
    # Map identities only, independently computed from game RNG; do not open a game or result from these seeds.
    sys_path=str(BASE)
    import sys
    sys.path.insert(0,sys_path)
    import map_identity
    blocked|={map_identity.local_board(s) for s in banned_seeds}
    official_parents={r["seed"] for r in prior["games"]}
    group=read(TOP/"manifest.json")["top5"];top={r["id"] for r in group}
    allrows=[]; games=set();boards=set();manifests={}
    for mf in [TOP/"manifest.json",*sorted(TOP.glob("appendix_*/manifest.json"))]:
        mp=mf.parent/"validation_maps.jsonl.gz"
        if not mp.exists():continue
        manifests[str(mf.relative_to(TOP))]=sha(mf)
        maps={r["id"]:r["initial_terrain_u8_sha256"] for r in map(json.loads,gzip.open(mp,"rt"))}
        for r in read(mf)["games"]:
            if not r.get("clean_terminal") or r["id"] not in maps:continue
            gid=r["id"];terrain=maps[gid];raw=REPLAYS/(gid+".json.gz")
            if gid in games or terrain in boards:continue
            if r["seed"] in banned_seeds|official_parents or terrain in blocked:continue
            candidates=[s for s,v in enumerate(r["seat_strategy_ids"]) if v in top]
            if not candidates:continue
            s=min(candidates,key=lambda i:key("subject",gid,i))
            def ident(i):
                v=r["seat_strategy_ids"][i]
                return "id:"+v if v else "name:"+r["seat_names"][i]
            allrows.append({"id":gid,"seed":r["seed"],"terrain":terrain,"raw_sha256":r["sha256"],"seat":s,
                            "generating_actor":ident(s),"opponent":ident(1-s),"opponent_name":r["seat_names"][1-s],
                            "cohort":"mutual" if len(candidates)==2 else "external","source":str(mf.relative_to(TOP)),
                            "raw_local":raw.exists()})
            games.add(gid);boards.add(terrain)
    inds={}
    for group in ("mutual","external"):
        names={r["opponent"] for r in allrows if r["cohort"]==group}
        inds[group]={n for j,n in enumerate(sorted(names,key=lambda n:key("blind-opponent-20261119",n))) if j%4==0}
    for r in allrows:
        r["split"]="new_opponent" if r["opponent"] in inds[r["cohort"]] else (
            "known_new_map" if int(key("new-terrain",r["terrain"])[:8],16)%5==0 else "train")
    train_names={r["opponent"] for r in allrows if r["split"]=="train"}
    for r in allrows:
        if r["split"]=="known_new_map" and r["opponent"] not in train_names:r["split"]="new_opponent"
    assert not train_names&{r["opponent"] for r in allrows if r["split"]=="new_opponent"}
    panel=[]
    for split in ("new_opponent","known_new_map"):
        for cohort,quota in (("mutual",8),("external",16)):
            pool=sorted([r for r in allrows if r["split"]==split and r["cohort"]==cohort and r["raw_local"]],
                        key=lambda r:key("public-A-on-foreign-maps",r["terrain"],r["id"]))
            seen=Counter();keep=[]
            for r in pool:
                if seen[r["opponent"]]<max(4,(quota+len({z["opponent"] for z in pool})-1)//max(1,len({z["opponent"] for z in pool}))):
                    keep.append(r);seen[r["opponent"]]+=1
                if len(keep)==quota:break
            panel.extend(keep)
    save(HERE/"splits.json.gz",{"maps":allrows,"manifests":manifests,"excluded_parent_maps":len(official_parents),
                                "excluded_fresh_map_ids":sorted(banned_seeds),"excluded_fresh_results_read":False})
    save(HERE/"panel.json",{"maps":panel,"split_sha256":sha(HERE/"splits.json.gz")})
    print(len(allrows),Counter(r["split"] for r in allrows),len(panel),Counter((r["cohort"],r["split"]) for r in panel))

if __name__=="__main__":main()
