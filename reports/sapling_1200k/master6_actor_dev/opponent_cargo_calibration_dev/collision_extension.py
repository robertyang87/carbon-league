"""Cash-flow labels for attack and defense including zero-cargo hunters; no changes to frozen primary targets."""
from collections import defaultdict
import gzip
import json
import numpy as np
from scipy.optimize import minimize
from data import HERE,ROOT,SOURCES,raw_ladder,FastCarbon,intent,sha
from model import FEATURES,sigmoid
from fit import load,logistic,equal,event_metrics

ET=('lose3','lose5','take3','take5')


def game(g,tracker):
    ladder=g['split']=='ladder';s=g['seat']
    if ladder:
        raw=raw_ladder(g);fs=raw['steps'];cfg=raw['configuration'];sim=FastCarbon(cfg).load_observation(fs[0][0]['observation'])
        cmds=[[v['action'] or {} for v in frame] for frame in fs[1:]]
    else:
        b=(ROOT/g['source']).read_bytes();assert sha(b)==g['sha256'];raw=json.loads(gzip.decompress(b));cfg=raw['game_configuration'];sim=FastCarbon(cfg).reset(seed=g['seed']);cmds=raw['actions']
    bytime=[]
    for t,jt in enumerate(cmds):
        ps=sim.obs(s)['players']; units={u:(side,*v) for side,pl in enumerate(ps) for u,v in pl[2].items()}
        arrival=defaultdict(list)
        for u,(side,p,c,role) in units.items():
            a=intent.MOVES.index(jt[side].get(u) or 'NONE');arrival[int(intent.DEST[p,a])].append(u)
        # Also locate a new hire if it wins immediately at its base; that UID did not yet exist at root.
        sim.step(*jt);losses={a[0]:a for a in sim.last_collisions};debits=defaultdict(float);credits=defaultdict(float)
        for u,loser,amount,received,winner in sim.last_collisions:
            if amount<=0 or winner is None or loser==winner:continue
            assert loser in (0,1) and winner==1-loser and received==winner and u in units
            side,p,c,role=units[u];a=intent.MOVES.index(jt[side].get(u) or 'NONE');cell=int(intent.DEST[p,a])
            win=[v for v in arrival[cell] if v not in losses and units[v][0]==winner and units[v][-1]=='COLLECTOR']
            if not win:
                # Before-move newly recruited workers are not present in the source observation.
                win=[v for v in sim.player_workers[winner] if sim.workers[v].pos==cell and sim.workers[v].is_collector]
            assert len(win)==1,(g['seed'],t,cell,u,win)
            v=win[0];debits[u]+=amount;credits[v]+=amount
            if losing_target:= (loser==1-s and role=='COLLECTOR' and c>0):
                tracker.append(dict(game=g['id'],observer_seat=s,t=t,victim=u,winner=v,amount=amount,
                                    hunter_pre_cargo=units[v][2] if v in units else 0.,cell=cell))
        assert abs(sum(debits.values())-sum(credits.values()))<1e-6
        bytime.append((debits,credits))
    return bytime


def main():
    data=load();cards=SOURCES['native']+SOURCES['ladder'];g=data['game'];times=data['t'];uids=data['uid'];x=data['x'].astype('d')
    sp=np.array([v['split'] for v in cards])[g];terrain=np.array([v['terrain'] for v in cards])[g]
    ev=np.zeros((len(g),4));tracker=[]
    for i,c in enumerate(cards):
        future=game(c,tracker)
        for j in np.flatnonzero(g==i):
            t=times[j];u=uids[j]
            ev[j]=[sum(e[0].get(u,0) for e in future[t:t+h]) for h in (3,5)]+[sum(e[1].get(u,0) for e in future[t:t+h]) for h in (3,5)]
    # Old exact gain bookkeeping is an independently derived check on the same winning workers.
    assert np.allclose(ev[:,2:],data['loot'],atol=1e-6)
    y=ev>0;tr=sp=='train';ca=sp=='calibration';wt=equal(terrain[tr]);wc=equal(terrain[ca]);mu=np.average(x[tr],axis=0,weights=wt)
    scale=np.maximum(.05,np.sqrt(np.average((x[tr]-mu)**2,axis=0,weights=wt)));z=np.clip((x-mu)/scale,-10,10)
    pri=[];w=[];bias=[];a=[];b=[];refs=[]
    for j,lab in enumerate(ET):
        beta,c=logistic(z[tr],y[tr,j],wt);w.append(beta);bias.append(c);lo=z[ca]@beta+c
        res=minimize(lambda k:np.average(np.logaddexp(0,k[0]*lo+k[1])-y[ca,j]*(k[0]*lo+k[1]),weights=wc),[1.,0.],bounds=[(0,10),(-20,20)])
        assert res.success,(j,res.message);a.append(res.x[0]);b.append(res.x[1]);pri.append(np.average(y[tr,j],weights=wt))
        v=FEATURES.index('lighter_enemy_nearest' if j<2 else 'heavier_enemy_nearest');r,k=logistic(z[tr,v,None],y[tr,j],wt);refs.append(sigmoid(z[:,v]*r[0]+k))
    w=np.stack(w,axis=1);pred=sigmoid((z@w+bias)*a+b);pred[:,1]=np.maximum(pred[:,0],pred[:,1]);pred[:,3]=np.maximum(pred[:,2],pred[:,3])
    np.savez_compressed(HERE/'collision_head.npz',mu=mu,scale=scale,weights=w,bias=bias,platt_slope=a,platt_bias=b)
    splits={'native_held':np.isin(sp,['recent_evaluation','held']),'ladder':sp=='ladder','train':tr,'calibration':ca}
    summary={}
    for name,mask in splits.items():
        weights=equal(terrain[mask]);d={}
        for j,lab in enumerate(ET):
            m=event_metrics(pred[mask,j],y[mask,j],weights);ref=event_metrics(refs[j][mask],y[mask,j],weights);prior=event_metrics(np.full(mask.sum(),pri[j]),y[mask,j],weights)
            d[lab]=dict(head=m,distance=ref,frequency=prior)
        codes={(c['id'],c['seat']) for i,c in enumerate(cards) if mask[g==i].any()}
        raw=[r for r in tracker if (r['game'],r['observer_seat']) in codes]
        summary[name]=dict(roots=int(mask.sum()),events=d,distinct_victim_transfers=len(raw),empty_hunter_visibly=int(sum(r['hunter_pre_cargo']==0 for r in raw)),
                           cash_taken_from_loaded_targets=float(sum(r['amount'] for r in raw)))
    (HERE/'collision_audit.json.gz').write_bytes(gzip.compress(json.dumps(dict(summary=summary,victim_transfers=tracker),separators=(',',':'),allow_nan=False).encode(),mtime=0))
    print('collisions',[(n,v['distinct_victim_transfers'],v['empty_hunter_visibly']) for n,v in summary.items()])


if __name__=='__main__':main()
