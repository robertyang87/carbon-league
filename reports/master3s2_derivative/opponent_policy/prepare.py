"""Recover each lecturer's real simultaneous order; compute only what the deploying seat can see."""
from pathlib import Path
import sys
import numpy as np
from original_policy import HERE,PROTOCOL,original,module,stream,need

def main(out):
    out.mkdir(parents=True,exist_ok=True)
    mod=module(original())
    exec((HERE/"runtime.py").read_text(),mod.__dict__)
    workers=mod._net()
    need(workers is not None,"Original full search network unavailable")
    for idx,c in enumerate(PROTOCOL["games"]):
        cached=out/(c["sha256"]+".npz")
        if cached.exists():
            with np.load(cached,allow_pickle=False) as z:
                need(str(z["source"])==c["sha256"] and int(z["seat"])==c["candidate_seat"],"Wrong cached source")
            continue
        chunks=[{k:[] for k in ("x","z","y","kind","t","pos","own_target","applied","truth_dst")} for _ in range(2)]
        for t,obs,cfg,xo,own,enemy in stream(c,mod):
            seat=c["candidate_seat"];op=1-seat
            lw,lb,_=workers(xo.astype(np.float32),mod._SID)
            players=obs["players"]
            us=players[seat][2]
            ally_destination={mod.move(unit[0],mod.WIDX[own.get(uid) or "NONE"]) for uid,unit in us.items()}
            for typ,(z,units,d) in enumerate(((lw,players[op][2],mod.WIDX),(lb,players[op][1],mod.BIDX))):
                uids=list(units)
                pos=[units[u][0] if typ==0 else units[u] for u in uids]
                if not pos:continue
                x=mod._opponent_features(xo,z,pos)
                for j,(uid,p) in enumerate(zip(uids,pos)):
                    a=d[enemy.get(uid) or "NONE"]
                    entry=chunks[typ];entry["x"].append(x[j]);entry["z"].append(z[0,p].copy());entry["y"].append(a)
                    entry["kind"].append(2 if typ else (0 if units[uid][2]=="COLLECTOR" else 1))
                    entry["t"].append(t);entry["pos"].append(p)
                    entry["truth_dst"].append(mod.move(p,a) if not typ else p)
                    entry["own_target"].append([int(mod.move(p,k) in ally_destination) for k in range(5)])
                    entry["applied"].append(uid)
        args={str(i)+k:np.asarray(vals) for i,chunk in enumerate(chunks) for k,vals in chunk.items()}
        np.savez(out/(c["sha256"]+".npz"),**args,source=c["sha256"],seat=c["candidate_seat"])
        if idx%12==0:print(idx+1,flush=True)
if __name__=="__main__":main(Path(sys.argv[1]))
