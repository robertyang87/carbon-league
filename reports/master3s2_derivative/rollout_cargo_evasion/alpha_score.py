"""Prior-exposed independent-response examples; no current opposing order is consumed by the predictor."""
from collections import Counter
import numpy as np
from scipy.special import softmax
from alpha import cards
from evaluate import arrays
from source import init
from register import H,sha,dump
from runtime import _ev_adjust,_ev_units

def main():
    m=init();params=arrays();out=[]
    for r in cards():
        p=r['players'];op=r['enemy'];pre=r['prev_players'];u=r['worker'];pos,cargo,role=p[op][2][u]
        lw,lb,_=m._RNET(r['x'][None],m._SID);lw=lw[0];lb=lb[0]
        if m._RNET is not m._NET:lw=lw/m._S_RTEMP;lb=lb/m._S_RTEMP
        units=_ev_units(p,op,pre);mask=next((mask for uid,_,_,mask in units if uid==u),None)
        temp=lw.copy()
        for uid,cell,*_ in units:temp[cell]*=params[0]
        child=_ev_adjust(p,op,pre,lw,params)
        choices=[];raw=[];target=r['label'];target=m.WIDX[target] if isinstance(target,str) else target
        for logits in [lw,temp,child]:
            y=softmax(logits[pos]);raw.append([float(v) for v in y])
            counter=Counter()
            for i in range(128):
                cmd=m._sample(p,op,logits,lb,np.random.RandomState(845331+i),r['step'])
                counter[m.WIDX[cmd.get(u,'NONE')]]+=1
            choices.append([counter[i] for i in range(5)])
        out.append(dict(stream=r['stream'],arm=r['arm'],step=r['step'],worker=u,position=pos,cargo=cargo,applies=mask is not None,
            detour_mask=None if mask is None else mask.tolist(),actual=int(target),actual_direction=['NONE','UP','RIGHT','DOWN','LEFT'][target],
            public_sha=r['public_sha256'],probabilities=raw,actual_propensities=[v[target] for v in raw],physical_counts=choices,
            physical_actual_probability=[v[target]/128 for v in choices]))
    print(dump(H/'alpha_scores.json',dict(weights_sha=sha((H/'weights.json').read_bytes()),rows=out,arms=['parent','temperature','evasion'],
         directions=['NONE','UP','RIGHT','DOWN','LEFT'],status='Previously exposed; two paired streams on one map, not four independent maps. Estimated enemy view from V4 actual public history, scored as a rollout-like continuation; no search.')))
    for r in out:
        if r['step']==269:print(r['stream'],r['arm'],r['actual_direction'],r['applies'],r['physical_actual_probability'])
if __name__=='__main__':main()
