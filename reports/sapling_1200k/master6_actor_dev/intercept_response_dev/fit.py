import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from forecast import carriers,_DIRECTION

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parent
sys.path.insert(0,str(ACTOR.parent/'all_top5'))
import fetch
sys.path.insert(0,str(ACTOR/'first_win_dev'))
import native

def take(players,seat,orders,rows,seed):
    for uid,_,_,x in carriers(players,seat):
        cmd=orders.get(uid)
        y=0 if cmd in (None,'NONE') else _DIRECTION.index(cmd)
        rows.append((x,y,seed))

def get():
    top=ACTOR.parent/'all_top5';records={}
    for p in [top/'manifest.json',*sorted(top.glob('appendix_*/manifest.json'))]:
        for r in json.loads(p.read_bytes())['games']:
            if 'claude-AlphaCarbon-Master6' in r['seat_names'] and r.get('frame_verified'):records[r['id']]=r
    panels=[[],[],[]]
    for key,r in sorted(records.items()):
        bits=hashlib.sha256(str(r['seed']).encode()).digest()
        if bits[0]%8:continue
        blob=(fetch.CACHE/'replays'/(key+'.json.gz')).read_bytes();assert fetch.sha(blob)==r['sha256']
        env,_,_=fetch.extract(blob)
        s=r['seat_names'].index('claude-AlphaCarbon-Master6')
        for before,after in zip(env['steps'],env['steps'][1:]):
            take(before[0]['observation']['players'],s,after[s].get('action') or {},panels[int(bits[1]%5==0)],r['seed'])
    for dirname in ['planter_escape_full_dev','planter_payback_full_dev','planter_capture_full_dev','capture_reach_dev']:
        for path in sorted((ACTOR/dirname/'games').glob('*/raw_*.json.gz')):
            raw=native.read(path);s=1-raw['candidate_seat']
            sim=native.exam.FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
            for commands in raw['actions']:
                take(sim.obs(1-s)['players'],s,commands[s],panels[2],raw['seed'])
                sim.step(*commands)
            assert list(sim.cash)==raw['official_score_by_seat']
    sets=[{r[2] for r in rows} for rows in panels]
    assert all(sets[i].isdisjoint(sets[j]) for i,j in [(0,1),(1,2),(0,2)])
    return panels

def arrays(rows):
    return np.stack([r[0] for r in rows]),np.array([r[1] for r in rows]),np.array([r[2] for r in rows])

def stats(w,x,y):
    z=x@w;z-=z.max(1,keepdims=True)
    ex=np.exp(z);p=ex/ex.sum(1,keepdims=True)
    return -np.log(p[np.arange(len(y)),y]),p

def main():
    train,held,transfer=map(arrays,get())
    x,y,_=train
    def obj(w):
        nll,p=stats(w,x,y);p[np.arange(len(y)),y]-=1
        return nll.sum()+2*(w@w),np.einsum('ijk,ij->k',x,p)+4*w
    sol=minimize(obj,np.zeros(x.shape[-1]),jac=True,method='L-BFGS-B',options=dict(maxiter=200))
    assert sol.success,sol.message
    # Equal penalty on a destination occupied by or reachable by a robber.
    baseline=np.array([0,1,0,-1,-1,0,0,0,0,0.])
    np.savez_compressed(HERE/'weights.npz',weights=sol.x,baseline=baseline)
    out={}
    for name,(xx,yy,ss) in zip(('train','held','native'),(train,held,transfer)):
        scores={}
        for model,weights in [('head',sol.x),('baseline',baseline)]:
            nll,p=stats(weights,xx,yy)
            scores[model]=dict(nll=float(nll.mean()),accuracy=float((p.argmax(1)==yy).mean()),
                               by_seed={str(int(s)):float(nll[ss==s].mean()) for s in np.unique(ss)})
        out[name]=dict(n=len(yy),maps=len(np.unique(ss)),scores=scores)
    out['admitted']=all(out[name]['scores']['head']['nll']<out[name]['scores']['baseline']['nll'] for name in ('held','native'))
    native.save(HERE/'evaluation.json',out)
    for name in ('held','native'):print(name,out[name]['n'],[(k,v['nll']) for k,v in out[name]['scores'].items()])
if __name__=='__main__':main()
