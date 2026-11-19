"""Input isolation: alter invisible native carbon and non-schema metadata on both held physical seats."""
import numpy as np
from gather import HERE,EXP,arena,read,sha,write,policy,FastCarbon

def main():
    panel=read(EXP/'panel.json');cards=[c for c in panel['cards'] if panel['split'][c['terrain']]=='test']
    sel=[next(c for c in cards if c['seat']==s) for s in (0,1)];tests=[]
    for c in sel:
        raw=read(arena.ROOT/c['path']);cfg=raw['game_configuration'];s=c['seat']
        models=[policy.load(EXP/'candidate.py') for _ in (0,1)]
        sim=FastCarbon(cfg).reset(seed=c['seed']);other=FastCarbon(cfg)
        for t,orders in enumerate(raw['actions'][:185]):
            view=sim.obs(s)
            if t in (0,48,114,183):
                st=sim.get_state();change=[i for i,v in enumerate(st['carbon']) if v>0 and view['carbon'][i]==0]
                for i in change:st['carbon'][i]=float(st['carbon'][i])+100.
                other.set_state(st);alt=other.obs(s);assert view['carbon']==alt['carbon']
                alt['full_carbon']=[1e6+i for i in range(225)]
                alt['next_opponent_action']={'impossible':'leak_marker'}
                ordinary=models[0].agent(view,cfg);mutated=models[1].agent(alt,cfg)
                assert ordinary==mutated and len(change)>0
                tests.append(dict(t=t,seat=s,changed_hidden_cells=len(change),matching_joint=True))
            else:
                for m in models:m._FEAT.observe(view)
            sim.step(*orders)
    write(HERE/'input_check.json',dict(cases=tests,gather_sha256=sha(HERE/'gather.py'),scope='Inputs to frozen actor and action encoder; synthetic private carbon and privileged extra fields stayed outside the public feature path.'))
if __name__=='__main__':main()
