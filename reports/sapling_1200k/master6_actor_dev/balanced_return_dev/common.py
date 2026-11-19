"""Restore a causal public history for both physical seats and score complete genuine proxy responses."""
import ast
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'blended_policy_return_dev'; BASE=HERE.parents[1]; ROOT=BASE.parents[1]
sys.path.insert(0,str(OLD))
import policy
from worlds import World, replay_official, CFG
PARENT=policy.PACK
IT50=ROOT/'data/weights/sapling_1200k/terminal_return_fit_v1.py'
TRACE=HERE.parent/'tree_response_training'


def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def encode(v):return json.dumps(v,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':')).encode()
def write(p,v):
    if p.exists():
        if read(p)!=v:raise ValueError('existing artifact differs '+str(p))
        return
    p.parent.mkdir(parents=True,exist_ok=True)
    d=encode(v)+b'\n';d=gzip.compress(d,mtime=0) if p.suffix=='.gz' else d
    tmp=p.with_name(p.name+'.partial');tmp.write_bytes(d);tmp.replace(p)
def net(name):
    path={'it20_self':PARENT,'frozen_it50':IT50}[name]
    assert sha(path)==read(HERE/'screen_protocol.json')['policies'][name]
    return policy.load(path)
def tapes():
    out={}
    for q in [TRACE/'source_receipt.json',*sorted(TRACE.glob('batches/intake_0[2345]/source_receipt.json'))]:
        for row in read(q)['games']:
            k=row['seed'],row['seat'];assert k not in out
            out[k]=ROOT/row['raw_path'],row['raw_sha256']
    return out
def open_state(seed,source_seat,t):
    path,digest=tapes()[seed,source_seat];assert sha(path)==digest
    data=read(path);w=World(seed)
    assert w.terrain_sha256==data['initial_terrain_u8_sha256']
    ff=[sys.modules['cnet2_feat'].Feat(s) for s in (0,1)]
    prefix=data['actions'][:t];assert len(prefix)==t
    for order in prefix:
        for s in (0,1):ff[s].observe(w.obs(s))
        assert w.advance(order) is None
    return w,ff,prefix,dict(path=str(path.relative_to(ROOT)),sha256=digest,terrain=w.terrain_sha256)
def decision(m,f,obs):
    x=f.observe(obs)
    return f.to_actions(m._mixed_actions(x,f.aux))
def verify(seed,prefix,suffix,w):
    a=replay_official(seed,prefix+suffix)
    for k in ('steps','terrain_sha256','extinct_by_seat','result_by_seat'):
        assert a[k]==w.terminal[k],k
    assert max(abs(a-b) for a,b in zip(a['scores'],w.terminal['scores']))<=.005001
    return a
def finish(seed,prefix,w,ff,nets,first=None):
    w=copy.deepcopy(w);ff=copy.deepcopy(ff);suffix=[]
    while not w.terminal:
        if first is not None and not suffix:
            joint=first
        else:
            joint=[decision(m,f,w.obs(i)) for i,(m,f) in enumerate(zip(nets,ff))]
        suffix.append(joint);w.advance(joint)
    return dict(actions=suffix,official=verify(seed,prefix,suffix,w))
def assert_cpu():
    assert os.sched_getaffinity(0)=={3}
    assert all(os.environ[k]=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'))
