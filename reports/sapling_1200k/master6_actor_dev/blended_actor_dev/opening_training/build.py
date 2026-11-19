"""Create an independently playable single-file release from the exact active actor."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'serving/candidate.py'
MATERIALIZED=Path('/workspace/sapling-opening-trained-agent.py')

def digest(data):
    return hashlib.sha256(data).hexdigest()

def build(out=MATERIALIZED):
    protocol=json.loads((HERE/'protocol.json').read_bytes())
    fit=json.loads((HERE/'fit.json').read_bytes())
    raw=BASE.read_bytes()
    if digest(raw)!=protocol['outside_step_zero_reference'] or digest((HERE/'opening_weights.npz').read_bytes())!=fit['weight_sha256']:
        raise ValueError('unexpected live/replacement weight identity')
    with np.load(HERE/'opening_weights.npz',allow_pickle=False) as a:
        weight,bias=a['weight'],a['bias']
    assert weight.ndim==2 and weight.shape[0]==3 and bias.shape==(3,)
    source=raw.decode()
    fn='policy.lin(hidden, "bhead")'
    assert source.count(fn)==2
    source=source.replace(fn,'_opening_base(policy, hidden)')
    payload='\n_OPENING_WT = np.array('+repr(weight.T.tolist())+',dtype=np.float32)\n_OPENING_B = np.array('+repr(bias.tolist())+',dtype=np.float32)\n'
    source+=payload+'''
_OPENING_ENABLED = True
_OPENING_FIRST_CALL = False
_opening_public_original = agent


def _opening_base(policy, hidden):
    out = policy.lin(hidden, "bhead")
    if _OPENING_FIRST_CALL and policy is _NETS[1]:
        return out + (hidden @ _OPENING_WT + _OPENING_B)
    return out


def agent(observation, configuration=None):
    global _OPENING_FIRST_CALL
    _OPENING_FIRST_CALL = _OPENING_ENABLED and int(observation["step"]) == 0
    try:
        return _opening_public_original(observation, configuration)
    finally:
        _OPENING_FIRST_CALL = False
'''
    data=source.encode();compile(source,str(out),'exec')
    assert len(data)<=10_000_000
    out.write_bytes(data)
    meta={'materialized_sha256':digest(data),'bytes':len(data),'base_sha256':digest(raw),'weights_sha256':fit['weight_sha256'],
          'build_sha256':digest(Path(__file__).read_bytes()),'architecture':'shared Sapling base head residual, enabled only at public step zero',
          'regeneration':'python build.py --out candidate.py','float_weights':int(weight.size+bias.size)}
    (HERE/'release.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=MATERIALIZED)
    print(build(p.parse_args().out))
