"""Check original and calibrated one-turn enemy orders, and every contested same-cell destination."""
import math
from collections import Counter
from pathlib import Path
import numpy as np
from original_policy import HERE,PROTOCOL,original,module,stream,need,dump,sha,build


def main():
    fixed=np.load(HERE/"weights.npz",allow_pickle=False)
    weights={k:fixed[k] for k in fixed}
    zero={k:(v if k.endswith("scale") else np.zeros_like(v)) for k,v in weights.items()}
    base=module(original())
    trained=module((HERE/"candidate.py").read_bytes())
    neutral=module(build(zero))
    for name in ("_WEIGHTS","_VALUE","_S_ABORT","_TEMP","_S_HOMEFIX","_SEED","_S_RTEMP","_S_GEN","_S_D2"):
        need(getattr(base,name)==getattr(trained,name)==getattr(neutral,name),"Different original runtime setting")
    need(not hasattr(trained,"_ROLL") and trained._RNET is trained._NET and base._RNET is base._NET,"Different deployed rollout")
    per=[];rows=[];maxroot=maxleaf=0.
    for c in PROTOCOL["games"]:
        if c["split"]!="val":continue
        counts=Counter()
        for t,obs,cfg,xo,own,other in stream(c,base):
            me=c["candidate_seat"];op=1-me;players=obs["players"]
            target=[tuple(xo.shape)]
            lw,lb,_=base._net()(xo.astype(np.float32),base._SID)
            raw_trained=trained._net()(xo.astype(np.float32),trained._SID)
            zlw,zlb=neutral._opponent_root_readout(xo,lw.copy(),lb.copy(),players,op)
            need(np.array_equal(lw,zlw) and np.array_equal(lb,zlb),"Zero residual has a different enemy logit")
            nw,nb=trained._opponent_root_readout(xo,lw.copy(),lb.copy(),players,op)
            need(np.array_equal(lw,raw_trained[0]) and np.array_equal(lb,raw_trained[1]),"Rollout policy changed")
            if counts["roots"]==0:
                sc=base._scalars(players,t,me)[None]
                vt=trained._net()(xo,trained._SID,sc)[2];vo=base._net()(xo,base._SID,sc)[2]
                maxleaf=max(maxleaf,float(np.max(abs(vt-vo))))
                root=base._POLICY.forward(xo[0],base._SID)
                new_root=trained._POLICY.forward(xo[0],trained._SID)
                maxroot=max(maxroot,max(float(abs(a-b).max()) for a,b in zip(root,new_root)))
            seed0=int(1000003*(t+1)+17*me)%(2**31)
            def predict(a,b):return base._sample(players,op,a[0],b[0],np.random.RandomState(seed0+7),t)
            a,b=predict(lw,lb),predict(nw,nb)
            need(a==predict(zlw,zlb),"Physical zero-head order drift")
            counts["roots"]+=1;counts["changed_joint"]+=int(a!=b)
            own_dst={base.move(u[0],base.WIDX[own.get(uid) or "NONE"]) for uid,u in players[me][2].items()}
            for uid,u in players[op][2].items():
                p,cargo,kind=u
                actual=base.move(p,base.WIDX[other.get(uid) or "NONE"])
                first=base.move(p,base.WIDX[a.get(uid) or "NONE"]);second=base.move(p,base.WIDX[b.get(uid) or "NONE"])
                counts["workers"]+=1;counts["actual_same_cell"]+=int(actual in own_dst)
                counts["old_arrival_hit"]+=int(first==actual);counts["new_arrival_hit"]+=int(second==actual)
                counts["old_collision_hit"]+=int(first in own_dst and actual in own_dst)
                counts["new_collision_hit"]+=int(second in own_dst and actual in own_dst)
                counts["old_collision_false_positive"]+=int(first in own_dst and actual not in own_dst)
                counts["new_collision_false_positive"]+=int(second in own_dst and actual not in own_dst)
            for uid in players[op][1]:
                ret=other.get(uid) or "NONE"; counts["bases"]+=1
                counts["old_base_hit"]+=int((a.get(uid) or "NONE")==ret);counts["new_base_hit"]+=int((b.get(uid) or "NONE")==ret)
            rows.append(dict(terrain=c["terrain"],seat=me,t=t,old=a,new=b,actual=other,own_actual=own))
        per.append(dict(seed=c["seed"],me=c["candidate_seat"],terrain=c["terrain"],counts=counts))
    totals=Counter()
    for p in per:totals.update(p["counts"])
    need(len(per)==8 and totals["roots"]==len(PROTOCOL["steps"])*8 and maxroot==maxleaf==0.,"Input/negative-control failure")
    dump(HERE/"held_orders.json.gz",rows)
    dump(HERE/"physical.json",dict(candidate_sha256=sha((HERE/"candidate.py").read_bytes()),max_own_root_logit_change=maxroot,
         max_unchanged_critic_change=maxleaf,zero_opponent_logits_bitwise_equal=True,all_original_rollout_logits_bitwise_equal=True,
         by_map_seat=per,totals=totals,orders_sha256=sha((HERE/"held_orders.json.gz").read_bytes()),
         scope="Same deterministic opponent root RNG as the original search. Collision means both units' issued destinations coincide before casualty resolution. Negative controls cover every held root opponent bilinears and representative per-seat own root and critic outputs."))
    print(totals)
if __name__=="__main__":main()
