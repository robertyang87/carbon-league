"""Exercise single-file public orders on real public history from both seats, compare exact signed actions."""
import argparse
import gzip
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
from data import D,ROOT,read,sha,compact,FastCarbon


def main(cache):
    reg=read(D/'source_registry.json');md=read(D/'package.json');assert sha(D/'candidate.py')==md['candidate_sha256']
    rows=[r for r in reg['games'] if r['split']=='held']
    spec=importlib.util.spec_from_file_location('joint_feature_public_package',D/'candidate.py');net=importlib.util.module_from_spec(spec);spec.loader.exec_module(net)
    look={}
    for r in read(D/'decisions.json.gz'):
        if r['model']=='trained' and r['split']=='held':look.setdefault((r['seed'],r['seat'],r['t']),[]).append(r)
    receipts=[]
    for i,g in enumerate(rows):
        if i not in (0,3):continue
        r=read(ROOT/g['source']);cfg=r['game_configuration'];s=r['candidate_seat'];sim=FastCarbon(cfg).reset(seed=r['seed'])
        z=np.load(cache/f'{i}.npz');pred=net._jrc_scores(z['x'],net._JRC_WEIGHTS);ptr=z['ptr'];roles=z['role']
        for t,orders in enumerate(r['actions']):
            k=(r['seed'],s,t);obs=sim.obs(s)
            if t==0 or t==298 or (t>=12 and (t-12)%16==0) or k in look:
                start=time.monotonic();chosen=net.agent(obs,cfg);wall=time.monotonic()-start;info=net._JRC_LAST
                lo,hi=ptr[t:t+2];win=lo+int(np.argmax(pred[lo:hi]));win=win if pred[win]-pred[lo]>=100 else lo
                assert chosen==info['chosen'] and info['parent']==orders[s]
                assert len(ptr)>t and info['menu']==hi-lo and info['role']==str(roles[win]) and abs(info['upgrade']-(pred[win]-pred[lo]))<.003
                assert (win!=lo)==(chosen!=orders[s]);assert all(chosen==n['chosen_command'] for n in look.get(k,[]))
                receipts.append(dict(seed=r['seed'],seat=s,t=t,known_signed_cases=len(look.get(k,[])),role=info['role'],upgrade=info['upgrade'],
                                     changed=chosen!=orders[s],parent=orders[s],chosen=chosen,wall_seconds=wall))
            else:net._FEAT.observe(obs)
            sim.step(*orders)
    assert len({r['seat'] for r in receipts})==2 and len({r['seed'] for r in receipts})==2
    out=dict(package_sha256=md['candidate_sha256'],roots=len(receipts),changes=sum(r['changed'] for r in receipts),observed_original_source_only=True,rows=receipts)
    (D/'package_parity.json.gz').write_bytes(gzip.compress(compact(out),mtime=0))
    print('single file',len(receipts),out['changes'])


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('cache',type=Path);main(a.parse_args().cache)
