"""Read back both served packs and exercise their greedy policy on complete observed histories."""
import base64
import gzip
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import zlib
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
W = ROOT / 'data/weights/sapling_1200k'
sys.path.insert(0, str(ROOT/'code'))
import cnet2_pack as builder


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(name):
    weight_path = W / (name+'.npz')
    pack_path = W / (name+'_agent.py' if name == 'cash_v3' else name+'.py')
    packed = pack_path.read_bytes()
    spec = importlib.util.spec_from_file_location(name, pack_path)
    pack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pack)
    if pack._OPTS != dict(teacher=2,shield=False,nocollide=False,endgame=False,pshield=False,rguard=False):
        raise ValueError('serving controls differ')
    with np.load(weight_path, allow_pickle=False) as raw:
        table, binary, _ = builder.quantize(raw)
        if table != pack._QMETA or zlib.decompress(base64.b64decode(pack._W64)) != binary or raw['meta'].item() != pack._WMETA:
            raise ValueError('pack did not embed the supplied actor')
    return pack,dict(pack=sha(packed), weights=sha(weight_path.read_bytes()), modules=pack._SRC_SHA256, options=pack._OPTS, tensors=len(table))


def main():
    modules, ids = zip(*(load(n) for n in ('cash_v3','master6_actor_v1')))
    if ids[0]['modules'] != ids[1]['modules'] or ids[0]['weights'] == ids[1]['weights']:
        raise ValueError('need identical controllers and different actors')
    split = json.loads((HERE/'data_split.json').read_bytes())['games']
    with gzip.open(HERE/'data_receipts.json.gz','rt') as f:
        steps = {g['id']: g['steps'] for g in json.load(f)['games']}
    pol = sys.modules['cnet2_np'].Policy
    original = pol.act
    calls = []
    def counted(self, x, aux, *args, **kwargs):
        if kwargs.get('teacher') != 2 or kwargs.get('hook') is not None or kwargs.get('sample_rng') is not None or kwargs.get('forced') is not None:
            raise ValueError('unexpected serving guard, sampling or teacher forcing')
        calls.append(id(self))
        return original(self, x, aux, *args, **kwargs)
    evidence = []
    pol.act = counted
    for game in (g for g in split if g['split']=='holdout'):
        path = Path('/workspace/sapling-all-top5-20261119/replays') / (game['id']+'.json.gz')
        raw = path.read_bytes()
        if sha(raw) != game['sha256']:
            raise ValueError('public trace identity')
        trace = json.loads(gzip.decompress(raw))
        seat = game['seat']
        # Earliest retained root in each fixed 75-step epoch. Selection is independent of actions.
        chosen = {min(s for s in steps[game['id']] if s//75==q) for q in set(s//75 for s in steps[game['id']])}
        states = []
        for t, frame in enumerate(trace['steps'][:max(chosen)+1]):
            shared=frame[0]['observation']
            obs={k:shared[k] for k in ('step','trees','players')}
            obs.update(player=seat,carbon=frame[seat]['observation']['carbon'])
            if t==0 or t in chosen:
                start=len(calls)
                actions=[m.agent(obs,trace['configuration']) for m in modules]
                if len(calls)!=start+2 or calls[-1]==calls[-2]:
                    raise RuntimeError('both independent served actors must run exactly once')
                if t==0:
                    states=[inspect.getclosurevars(m._AG).nonlocals['st']['f'] for m in modules]
                    if len({id(s) for s in states})!=2:
                        raise RuntimeError('shared feature history')
                if t in chosen:
                    u,y=states[0], states[0].labels(trace['steps'][t+1][seat]['action'])
                    a,b=[u.labels(x) for x in actions]
                    mask=y>=0
                    evidence.append(dict(id=game['id'],seat=seat,step=t,active=int(mask.sum()),changed=int(((a!=b)&mask).sum()),
                                         expert_correct_parent=int(((a==y)&mask).sum()),expert_correct_candidate=int(((b==y)&mask).sum()),
                                         two_neural_calls=True))
            else:
                # Only Feat stores history in this stateless actor. Advance every physical public
                # observation so the normal pack entry sees the full history at each sampled root.
                for f in states:
                    f.observe(obs)
    pol.act=original
    out=dict(check='exact published actor tensors and identical serving control; both physical seats, one autonomous actual pack call per tested snapshot per actor',
             denominator='all occupied command slots in sampled roots, including forced actions; no simulated next state',
             pack_identities=ids,roots=len(evidence),maps=len({e['id'] for e in evidence}),
             changed_roots=sum(e['changed']>0 for e in evidence),sums={k:sum(e[k] for e in evidence) for k in ('active','changed','expert_correct_parent','expert_correct_candidate')},
             by_phase={str(q):sum(1 for e in evidence if e['step']//75==q) for q in range(4)},examples=evidence)
    p=HERE/'serving.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
    print({k:v for k,v in out.items() if k not in ('examples','pack_identities')})

if __name__=='__main__':main()
