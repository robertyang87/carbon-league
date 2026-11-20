"""Run the released embedded _turn with the official hidden board and the two issued actions."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time
import subprocess

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parents[1]
PACK=FAMILY/"cold_start/candidate.py"
SHA="7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed"
SNAP="805f6ab04fd81959bf43fd1db00d1d7724ffa5e5082c7a1251e6c6405bf2b338"
TOL=1e-8


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def extract(raw):
    src=raw.decode().replace("\r\n","\n")
    native=src[src.index("from collections import defaultdict\n\nMOVES ="):src.index("import time as _time\n",src.index("from collections import defaultdict\n\nMOVES ="))]
    turn=src[src.index("def _turn(o, configuration, acts):"):src.index("_S_START =")]
    return native,turn


def engine():
    native,turn=extract(PACK.read_bytes())
    exports=dict(_RX=True,_RX_CFG=[None,None])
    exec(compile(native+"\n"+turn,"pinned_embedded_turn","exec"),exports)
    return exports["_turn"]


def compare(got,actual,meta):
    flags=set();ex={};err={};physical=set()
    def flag(k,x,y,real=True):
        flags.add(k);ex.setdefault(k,[x,y])
        if real: physical.add(k)
    def money(k,x,y):
        diff=abs(x-y);err[k]=max(err.get(k,0.),diff)
        if diff>TOL:flag(k,x,y)
    for s,(a,b) in enumerate(zip(got["players"],actual["players"])):
        money("cash",a[0],b[0])
        if a[1]!=b[1]:flag("base",a[1],b[1])
        for ix,kind in ((2,"worker"),(3,"tree")):
            left={v[0]:(uid,v[1:]) for uid,v in a[ix].items()};right={v[0]:(uid,v[1:]) for uid,v in b[ix].items()}
            if set(left)!=set(right):
                flag(kind+"_position",sorted(set(left)-set(right)),sorted(set(right)-set(left)))
            for pos in left.keys()&right.keys():
                uid,va=left[pos];vid,vb=right[pos]
                if uid!=vid:flag(kind+"_id",uid,vid,False)
                money("worker_cargo" if kind=="worker" else "tree_age",va[0],vb[0])
                if ix==2 and va[1]!=vb[1]:
                    flag("worker_role",va[1],vb[1])
                if ix==3:
                    cm,cn=meta[uid],actual["trees"][vid]
                    if cm[0]!=cn[0]:flag("creator_id",cm[0],cn[0],False)
                    money("tree_income",cm[1],cn[1])
    for a,b in zip(got["carbon"],actual["full_carbon"],strict=True):
        money("carbon",a,b)
    return flags,ex,err,physical


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--original",type=Path)
    opts=parser.parse_args()
    orig=(opts.original.read_bytes() if opts.original else
          subprocess.check_output(["git","-C",str(FAMILY.parents[1]),"show","HEAD:sources/master3s2/master3s2.py"]))
    candidate=PACK.read_bytes()
    assert sha(candidate)==SHA and sha(orig)=="65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
    assert extract(orig)==extract(candidate)
    segments=dict(zip(("embedded_engine","turn_wrapper"),(sha(s.encode()) for s in extract(orig))))
    snapshot=(FAMILY/"ladder/v1_placement_review/snapshot.json").read_bytes()
    assert sha(snapshot)==SNAP
    snap=json.loads(snapshot)
    turn=engine(); start=time.perf_counter()
    all_counts=Counter();maxima={};first={};by_game=[];detail=[]
    for card in snap["official_matches_unfiltered"].values():
        if card["kind"]!="newrank-placement" or not card["rated"] or card["state"]!="done":continue
        for g in card["result"]:
            key=card["id"]+"-"+str(g["game"])
            raw=(FAMILY/"ladder/replays"/(key+".json.gz")).read_bytes()
            env=json.loads(gzip.decompress(raw)); cfg=env["configuration"];frames=env["steps"];counts=Counter();persistent=None
            assert len(frames)==300 and env["statuses"]==["DONE","DONE"]
            for t,(before,after) in enumerate(zip(frames[:-1],frames[1:],strict=True)):
                root=before[0]["observation"]
                full={k:root[k] for k in ("players","trees","step")}
                full["carbon"]=root["full_carbon"]
                actions=[s["action"] for s in after]; nxt=after[0]["observation"]
                got=turn(full,cfg,actions)
                persistent=turn(full if persistent is None else persistent,cfg,actions)
                hist,_,_,hist_phys=compare(persistent,nxt,persistent["_st"].trees_meta)
                if hist:
                    counts["persistent_any_mismatch"]+=1
                    if hist_phys:
                        counts["persistent_physical_mismatch"]+=1
                    for field in hist:
                        counts["persistent_"+field]+=1
                assert got["step"]==nxt["step"]==t+1
                flags,ex,err,physical=compare(got,nxt,got["_st"].trees_meta)
                counts["steps"]+=1
                for k,v in err.items():maxima[k]=max(maxima.get(k,0.),v)
                if physical:counts["physical_mismatch"]+=1
                if flags:
                    counts["any_mismatch"]+=1
                    if not physical:counts["only_ids"]+=1
                    for k in flags:first.setdefault(k,dict(game=key,t=t,example=ex[k]))
                    counts.update(flags)
                    detail.append(dict(game=key,t=t,flags=sorted(flags),physical=sorted(physical),example=ex))
            by_game.append(dict(game=key,sha256=sha(raw),counts=counts))
            all_counts.update(counts)
    assert len(by_game)==32 and all_counts["steps"]==9568
    for k in ("persistent_any_mismatch","persistent_physical_mismatch","physical_mismatch"):
        all_counts.setdefault(k,0)
    run=dict(snapshot=SNAP,candidate_sha=SHA,original_sha=sha(orig),same_source=segments,tol=TOL,counts=all_counts,first_differences=first,
             max_abs_error=maxima,by_game=by_game,differences=detail)
    packed=gzip.compress(json.dumps(run,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode(),mtime=0)
    (HERE/"result.json.gz").write_bytes(packed)
    print(dict(all_counts),maxima,sha(packed),round(time.perf_counter()-start,1),flush=True)


if __name__=="__main__":
    main()
