"""Build the 75% Echo4 + unchanged 25% Sapling research agent by replacing only the expert payload."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = BASE.parents[1]
PARENT = ROOT/'data/weights/sapling_1200k/terminal_return_fit_v1.py'
STUDENT = ROOT/'data/weights/sapling_1200k/master6_actor_v1.npz'
ECHO = ROOT/'code/packs/echo4_int8.py'
IT50 = ROOT/'code/packs/ppou2_m_it50.py'
FROZEN = 'b2069da2f42a18c3e0e0296d2e16e45b3681d7cc'
EXPERT = ('_B1','_Q1','_M1')
STUDENT_KEYS=('_W64','_QMETA','_WMETA')
CAND = HERE/'candidate.py'


def digest(p):
    return hashlib.sha256(p.read_bytes() if isinstance(p,Path) else p).hexdigest()


def seal(p,data):
    if p.exists():
        if p.read_bytes()!=data:raise ValueError('existing data differ: '+str(p))
    else:
        temp=p.with_suffix('.partial');temp.write_bytes(data);temp.replace(p)


def out(p,d):
    seal(p,(json.dumps(d,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())


def input_bytes(path):
    raw=path.read_bytes()
    remote=subprocess.check_output(['git','show',f'{FROZEN}:{path.relative_to(ROOT)}'],cwd=ROOT)
    if raw!=remote:raise ValueError(f'working input differs: {path}')
    return raw


def constants(text,names):
    nodes = {n.targets[0].id:n for n in ast.parse(text).body if isinstance(n,ast.Assign)
             and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in names}
    if set(nodes)!=set(names):raise ValueError('incomplete pack')
    return nodes


def payload(text,names):
    return {k:ast.literal_eval(n.value) for k,n in constants(text,names).items()}


def compact(v):
    return json.dumps(v,sort_keys=True,ensure_ascii=True,separators=(',',':')).encode()


def build():
    old = input_bytes(PARENT).decode(); spec = input_bytes(ECHO).decode()
    e50 = input_bytes(IT50).decode()
    if digest(old.encode())!='9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0' or payload(old,EXPERT)!=payload(e50,EXPERT):
        raise ValueError('parent expert is not unchanged it50')
    if digest(input_bytes(STUDENT))!='f82bfd47a904aa1fbfa4e95fbfcdd13e623c1f1eb216a8e736694a1d282cc713':
        raise ValueError('student training weights changed')
    lines=old.splitlines(keepends=True); assets=payload(spec,EXPERT)
    for name,node in sorted(constants(old,EXPERT).items(),key=lambda kv:-kv[1].lineno):
        lines[node.lineno-1:node.end_lineno]=[f'{name} = {assets[name]!r}\n']
    new=''.join(lines)
    new=new.replace('"expert": "ppou2_m_it50"','"expert": "Echo4_it20"')
    if payload(new,EXPERT)!=assets or payload(new,STUDENT_KEYS)!=payload(old,STUDENT_KEYS) or new==old or len(new.encode())>=10000000:
        raise ValueError('actor identity or package limit')
    conf=payload(new,['_HYBRID'])['_HYBRID']
    if (conf['expert'],conf['expert_weight'],conf['student_weight'],conf['guard'],conf['search'])!=('Echo4_it20',.75,.25,False,False):
        raise ValueError('runtime config drift')
    seal(CAND,new.encode())
    manifest=dict(frozen_main=FROZEN,input_sha256={str(p.relative_to(ROOT)):digest(p) for p in [PARENT,ECHO,IT50,STUDENT]},
        candidate_sha256=digest(CAND),package_bytes=CAND.stat().st_size,expert_sha256=digest(compact(assets)),
        student_quantized_sha256=digest(compact(payload(new,STUDENT_KEYS))),previous_expert_sha256=digest(compact(payload(old,EXPERT))),
        network_config=conf,build_sha256=digest(Path(__file__)),protocol_sha256=digest(HERE/'protocol.json'))
    out(HERE/'release.json',manifest)
    return manifest


if __name__=='__main__':
    print(build())
