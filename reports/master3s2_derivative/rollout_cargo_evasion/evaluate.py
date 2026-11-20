"""Score genuinely issued opponent commands and native simultaneous team samples on held terrain."""
from collections import defaultdict
import gzip
import json
import numpy as np
from scipy.special import logsumexp

from source import collect,init
from register import H,sha,read,dump
from runtime import _ev_adjust
from fit import PROTOCOL

ARMS=('parent','temp','learned')
def arrays():
    spec=read(H/'weights.json')
    assert sha((H/'weights.json').read_bytes())==read(H/'fit.json')['weights']
    return (spec['beta'],*[np.array(spec[k],np.float32) for k in ['w','b','out']])

def sample(m,root,params):
    seat=1-root['seat'];ps=root['players'];prev=root['previous'];lw=root['lw']
    units={w[0]:w for w in root['units']};alluids=list(root['labels'])
    temp=lw.copy()
    for u,pos,x,det in units.values():temp[pos]*=params[0]
    learned=_ev_adjust(ps,seat,prev,lw,params)
    masks={u:x[3] for u,x in units.items()}
    counts=np.zeros((3,len(alluids),5),np.int16);full=np.zeros(3,np.int16)
    game,step=root['game'],root['step']
    first=int(sha(f'rollout_evasion/{game}/{step}/{seat}'.encode())[:8],16)&0x7fffffff
    ord_ok=0
    original_team=None
    for k in range(24):
        orig=None
        for j,logs in enumerate((lw,temp,learned)):
            if j>0 and not units: cmd=orig
            else:cmd=m._sample(ps,seat,logs,root['lb'],np.random.RandomState(first+k),step)
            if j==0:orig=cmd
            for i,u in enumerate(alluids):counts[j,i,m.WIDX[cmd.get(u,'NONE')]]+=1
            # Only compare worker commands to known labels here; base labels are not collected.
            full[j]+=all(m.WIDX[cmd.get(u,'NONE')]==root['labels'][u] for u in alluids)
        if not units:
            assert learned is lw and np.array_equal(lw,temp)
    out=[]
    for i,u in enumerate(alluids):
        y=root['labels'][u];pos=ps[seat][2][u][0];q=counts[:,i,:].astype(float)/24.
        raw=[];det=[];det_direct=[]
        for v in (lw,temp,learned):
            l=v[pos].astype(np.float64)
            raw.append(float(logsumexp(l)-l[y]))
            if u in units:det_direct.append(float(np.exp(l-logsumexp(l))[masks[u]].sum()))
        expected=np.zeros(5);expected[y]=1.
        brier=((q-expected)**2).sum(1)
        d=dict(uid=u,role=ps[seat][2][u][2],cargo=ps[seat][2][u][1],actual=y,eligible=u in units,
               root_active=bool(units),raw_nll=raw,physical_brier=brier.tolist(),prob_actual=q[:,y].tolist())
        if u in units:
            mask=masks[u];det=q[:,mask].sum(1);event=bool(mask[y])
            d.update(detour_mask=mask.tolist(),actual_detour=event,raw_detour_probability=det_direct,detour_probability=det.tolist(),
                detour_brier=((det-float(event))**2).tolist(),detour_raw_brier=((np.array(det_direct)-event)**2).tolist())
        out.append(d)
    return dict(seed=root['seed'],terrain=root['terrain'],seat=root['seat'],step=step,game=game,arm=root['arm'],
                full_worker_joint_matches=full.tolist(),active=bool(units),workers=out)

def summary(rows,kind):
    by=defaultdict(list);div={}
    for r in rows:
        workers=[w for w in r['workers'] if (w['eligible'] if kind=='eligible' else (not w['eligible'] and (r['active'] if kind=='cooccurring' else not r['active'])))]
        by[r['terrain']].extend(workers)
    for key,ws in by.items():
        if not ws:continue
        metrics=['raw_nll','physical_brier']+(['detour_brier','detour_raw_brier','detour_probability'] if kind=='eligible' else [])
        div[key]=dict(workers=len(ws),actual_detours=sum(u.get('actual_detour',False) for u in ws),
            metrics={n:np.mean([u[n] for u in ws],axis=0).tolist() for n in metrics})
    out={k:np.mean([v['metrics'][k] for v in div.values()],axis=0).tolist() for k in next(iter(div.values()))['metrics']} if div else {}
    return dict(workers=sum(c['workers'] for c in div.values()),maps=len(div),by_terrain=div,map_equal=out)

def main():
    m=init();packed=arrays(); tickets=read(H/'protocol.json')['held'];roots,seen=collect(m,tickets,True)
    data=[sample(m,r,packed) for r in roots]
    b=(json.dumps(data,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
    path=H/'held_rows.json.gz';path.write_bytes(gzip.compress(b,mtime=0))
    arms={}
    for name,arm in [('v4','challenger'),('v2','parent')]:
        sub=[r for r in data if r['arm']==arm]
        sums={k:summary(sub,k) for k in ['eligible','ordinary','cooccurring']}
        sums['n_active']=sum(r['active'] for r in sub)
        sums['n_ordinary_sampled']=sum(not r['active'] for r in sub)
        sums['eligible_event_counts']=[sum(w['actual_detour'] for r in sub for w in r['workers'] if w['eligible']),sums['eligible']['workers']]
        eligible=sums['eligible'];score=eligible['map_equal'];ordinary=sums['ordinary']['map_equal']
        no_regression_maps=sum(v['metrics']['detour_brier'][2]<=v['metrics']['detour_brier'][1] for v in eligible['by_terrain'].values())
        checks=dict(minimum=eligible['maps']>=3 and eligible['workers']>=50,detour_better_than_parents=score['detour_brier'][2]<min(score['detour_brier'][:2]),
             three_nonworse_temperature_maps=no_regression_maps>=3,categorical_no_worse=score['physical_brier'][2]<=score['physical_brier'][0],
             no_ordinary_drift=max(ordinary['physical_brier'])==min(ordinary['physical_brier']))
        sums['checks']=checks;sums['pass_gate']=all(checks.values());arms[name]=sums
    print(dump(H/'held.json',dict(protocol=PROTO(),fit=sha((H/'fit.json').read_bytes()),weights=sha((H/'weights.json').read_bytes()),
       rows=sha(path.read_bytes()),samples=seen,estimators=list(ARMS),domain=arms)),json.dumps({k:[v['eligible_event_counts'],v['eligible']['map_equal'],v['checks']] for k,v in arms.items()}),flush=True)
def PROTO():
    digest=sha((H/'protocol.json').read_bytes())
    assert digest==PROTOCOL
    return digest
if __name__=='__main__':main()
