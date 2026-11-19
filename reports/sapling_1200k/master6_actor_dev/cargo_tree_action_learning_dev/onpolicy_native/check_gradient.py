"""Differentiate the actual probability returned by three independent public training agents."""
import os
import numpy as np
from experiment import HERE, ROOT, FastCarbon, build, model, read, sha, write

def run():
    protocol=read(HERE/'protocol.json')
    rng=np.random.default_rng(611983)
    direction=rng.normal(size=(97,5)).astype(np.float32)
    direction/=np.linalg.norm(direction)
    eps=.01
    out=[]
    os.environ.pop('SAPLING_NATIVE_TRACE',None)
    for c in protocol['historical_controls'][:2]:
        raw=read(ROOT/c['raw']);s=c['seat'];seed=c['seed']
        os.environ['SAPLING_NATIVE_STREAM']=str(seed)
        sim=FastCarbon(raw['game_configuration']).reset(seed=seed)
        models=[model(build()) for _ in range(3)]
        for t,joint in enumerate(raw['actions'][:85]):
            obs=sim.obs(s)
            if t in (0,12,48,84):
                for m in models:m.agent(obs,sim.config)
                if t==0:
                    for sign,m in zip((0,+1,-1),models):
                        m._NETS[1].w['whead.weightT']+=sign*eps*direction[:-1]
                        m._NETS[1].w['whead.bias']+=sign*eps*direction[-1]
                else:
                    ids=[];ev=[m.last_training_event for m in models]
                    for e in ev:ids.append((e['slot'],e['route'],e['prefix'],e['actions'],e['options']) if e['supported'] else None)
                    smooth=bool(ids[0] is not None and ids[0]==ids[1]==ids[2])
                    z=dict(seed=seed,seat=s,t=t,same_menu_and_prefix=smooth)
                    if smooth:
                        base=ev[0];a=base['actions'];j=base['choice'];p=base['probabilities']
                        dz=.25*np.r_[base['hidden'],1.]@direction[:,a]
                        slope=float(dz[j]-np.dot(p,dz))
                        apparent=float((np.log(ev[1]['probabilities'][j])-np.log(ev[2]['probabilities'][j]))/(2*eps))
                        if abs(apparent-slope)>3.e-4:raise ValueError('actual distribution score gradient differs')
                        z.update(analytic=slope,measured=apparent,error=abs(apparent-slope))
                    out.append(z)
            else:
                for m in models:m._FEAT.observe(obs)
            sim.step(*joint)
    if sum(v['same_menu_and_prefix'] for v in out)<2:raise ValueError('insufficient smooth samples')
    write(HERE/'score_gradient.json',dict(program=sha(HERE/'check_gradient.py'),protocol=sha(HERE/'protocol.json'),eps=eps,rows=out))
    print(out,flush=True)
if __name__=='__main__':run()
