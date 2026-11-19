"""Build planter directional decisions on the actual demonstrator prefix, from training and held maps."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import gzip
import hashlib
import inspect
import json
import os
import sys
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
SEQ=BASE/"territory_sequence"
sys.path.insert(0,str(SEQ))
import data as terr
from gather import policy

read=terr.read
write=terr.write
sha=terr.sha
PARENT=BASE/"candidate.py"
RULE=read(HERE/"protocol.json")


def collect(shard):
    n,cs=shard
    os.sched_setaffinity(0,{2+n})
    parent=policy.load(PARENT)
    marker="action = int(np.argmax(np.where(legal, combined, -np.inf)))"
    src=inspect.getsource(parent._return_decode)
    assert src.count(marker)==1
    buf={}
    def get(slot,hidden,move,z,legal,chosen):
        target=int(labels[slot])
        if slot<10:
            assert legal[target]
            buf[slot]=(hidden.copy(),move.copy(),z.copy())
        return target if legal[target] else chosen
    ns=dict(vars(parent),get=get)
    exec(src.replace(marker,marker+"\n        action = get(slot,states[1][3],move,combined,legal,action)"),ns)
    decoder=ns["_return_decode"]
    rows=[]; hs=[]; ms=[]; zs=[]
    sees=0
    for c in cs:
        cfg,actions,scores,full=terr.load_game(c)
        fresh=lambda:terr.FastCarbon(cfg).reset(seed=c["seed"],full_carbon=full)
        sim=fresh(); local,bank=terr.credits(sim,actions,c["seat"])
        sim=fresh(); F=sys.modules["cnet2_feat"]; feat=F.Feat(c["seat"])
        s=c["seat"]
        for t,joint in enumerate(actions):
            obs=sim.obs(s)
            x=feat.observe(obs)
            if 64<=t<=192 and t%4==0 and t+24<=len(actions):
                labels=feat.labels(joint[s]); uses=[(k,u) for k,u in enumerate(feat.own_uid[:10])
                    if u is not None and x["tok_kind"][k]==1 and labels[k]>0]
                if uses:
                    enc=[]
                    for net in parent._NETS:
                        toks,ctx=net.encode(x,2);enc.append((net,toks,ctx,net.lin(ctx,"h0")))
                    buf.clear(); decoder(x,feat.aux,enc,False);sees+=1
                    for k,u in uses:
                        h,m,z=buf[k];pos=obs["players"][s][2][u][0];act=joint[s][u]
                        dest=int(terr.TO[pos,terr.DIRECTIONS.index(act)])
                        ar=terr.D[dest]<=2
                        y=float(local[t:t+24,ar].sum()/24-local[t-8:t,ar].sum()/8+.25*bank[t:t+24].sum()/(24*len(obs["players"][s][2])))
                        opp=obs["players"][1-s][2]
                        before=min((int(terr.D[pos,w[0]]) for w in opp.values()),default=15)
                        after=min((int(terr.D[dest,w[0]]) for w in opp.values()),default=15)
                        foe_dist=(before,after)
                        rows.append(dict(map=c["terrain"],game=c["sha256"],source=c["source"],split=c["split"],teacher=c["teacher"],
                            seat=s,seed=c["seed"],t=t,uid=u,slot=k,label=int(labels[k]),action=act,return24=y,nplanters=len(uses),
                            before_enemy=foe_dist[0],after_enemy=foe_dist[1],eventual_margin=float(scores[s]-scores[1-s])))
                        hs.append(h);ms.append(m);zs.append(z)
            sim.step(*joint)
        print(n,c["seed"],len(rows),flush=True)
    return rows,hs,ms,zs,sees


def main():
    assert sha(PARENT)==RULE["parent"] and sha(SEQ/"sources.json")==RULE["corpus"]
    cs=read(SEQ/"sources.json")
    assert all(c["seed"] not in set(range(621758343,621758354))|{621758326,621758328,621758330,621758339} for c in cs)
    with ProcessPoolExecutor(2) as pool:
        p,q=pool.map(collect,[(i,cs[i::2]) for i in range(2)])
    rows=p[0]+q[0]
    for a,b in [("train","test"),("train","val"),("val","test")]:
        assert not ({r["map"] for r in rows if r["split"]==a}&{r["map"] for r in rows if r["split"]==b})
    np.savez_compressed(HERE/"features.npz",h=np.stack(p[1]+q[1]),m=np.stack(p[2]+q[2]),z=np.stack(p[3]+q[3]))
    write(HERE/"rows.json.gz",rows)
    write(HERE/"data.json",dict(views=p[4]+q[4],decisions=len(rows),inputs=sha(HERE/"features.npz"),rows=sha(HERE/"rows.json.gz"),
        horizons=["t64–192 every 4","24-turn observed local economics"],splits={k:{"maps":len({r["map"] for r in rows if r["split"]==k}),
        "decisions":sum(r["split"]==k for r in rows),"losing_map_decisions":sum(r["split"]==k and r["eventual_margin"]<0 for r in rows),
        "moving_farther_from_nearest_rival":sum(r["split"]==k and r["after_enemy"]>r["before_enemy"] for r in rows)}for k in ("train","val","test")}))

if __name__=="__main__": main()
