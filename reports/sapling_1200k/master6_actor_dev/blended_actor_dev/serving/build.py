"""Replace only the embedded quantized Sapling tensors in the active two-network package."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib
import numpy as np

HERE=Path(__file__).resolve().parent
DEV=HERE.parent
ACTOR=DEV.parent
ROOT=ACTOR.parents[2]
BASE=ACTOR/'actor_partner_dev/candidate.py'
BASE_SHA='c5ad3f43959a4812e84e89574ee0c6e69765ebb06a8b3ec93f839b0dde128dd3'
SOURCE=ROOT/'data/weights/sapling_1200k/master6_actor_v1.npz'
SOURCE_SHA='f82bfd47a904aa1fbfa4e95fbfcdd13e623c1f1eb216a8e736694a1d282cc713'
sys.path.insert(0,str(ROOT/'code'))
import cnet2_pack
STUDENT=('_W64','_QMETA','_WMETA')


def check(test,message):
    if not test:raise ValueError(message)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def encode(v):
    return json.dumps(v,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':')).encode()


def save(p,raw):
    if p.exists() and p.read_bytes()==raw:return
    temp=p.with_name(p.name+'.new');temp.write_bytes(raw);temp.replace(p)


def fields(src,names):
    nodes={}
    for n in ast.parse(src).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            k=n.targets[0].id
            if k in names:
                check(k not in nodes,'duplicate '+k);nodes[k]=n
    check(set(nodes)==set(names),'missing assigned field')
    return nodes


def literal(src,names):
    return {k:ast.literal_eval(n.value) for k,n in fields(src,names).items()}


def substitute(src,values):
    lines=src.splitlines(keepends=True)
    nodes=fields(src,values)
    for k,n in sorted(nodes.items(),key=lambda q:-q[1].lineno):
        lines[n.lineno-1:n.end_lineno]=[f'{k} = {values[k]!r}\n']
    return ''.join(lines)


def quantize(path,reference):
    check(cnet2_pack.QMODE=='int8','requires production int8 packer')
    with np.load(path,allow_pickle=False) as z:
        check(set(z.files)==set(reference.files),'missing or extra model weights')
        cfg=json.loads(str(z['meta']))['cfg'];check(cfg==json.loads(str(reference['meta']))['cfg'],'architecture mismatch')
        for k in z.files:
            if k!='meta':
                check(z[k].shape==reference[k].shape and np.isfinite(z[k]).all(),'shape or finiteness of '+k)
        meta,raw,decoded=cnet2_pack.quantize(z)
        cfgmeta=str(z['meta'])
    check(all(np.isfinite(v).all() for v in decoded.values()),'nonfinite quantized tensor')
    return dict(_W64=base64.b64encode(zlib.compress(raw,9)).decode(),_WMETA=cfgmeta,_QMETA=meta)


def validate_baseline():
    check(sha(BASE)==BASE_SHA and sha(SOURCE)==SOURCE_SHA,'frozen deployed actor or trained initial state changed')
    source=BASE.read_text();old=literal(source,STUDENT)
    with np.load(SOURCE,allow_pickle=False) as ref: p=quantize(SOURCE,ref)
    check(p==old,'public actor and quantizer failed exact parent reconstruction')
    net=literal(source,('_HYBRID','_B1','_Q1','_M1'))
    expert=hashlib.sha256(encode({k:net[k] for k in ('_B1','_Q1','_M1')})).hexdigest()
    check(net['_HYBRID']['expert_weight']==.75 and net['_HYBRID']['student_weight']==.25 and net['_HYBRID']['style']==2,
          'changed active mixture')
    return source,old,expert


def build():
    source,old,expert=validate_baseline()
    proof={}
    with np.load(SOURCE,allow_pickle=False) as ref:
        for name in ('control','candidate'):
            path=DEV/(name+'.npz')
            values=quantize(path,ref)
            output=substitute(source,values)
            check(literal(output,STUDENT)==values and substitute(output,old)==substitute(source,old),'changed nonstudent implementation')
            dst=HERE/(name+'.py')
            check(len(output.encode())<=10_000_000,'single-file size')
            save(dst,output.encode())
            proof[name]={'npz_sha256':sha(path),'deployed_sha256':sha(dst),'bytes':dst.stat().st_size,
                         'embedded_int8_sapling_sha256':hashlib.sha256(encode(values)).hexdigest()}
    record={'base_pack_sha256':BASE_SHA,'parent_npz_sha256':SOURCE_SHA,'expert_payload_sha256':expert,
            'parent_quantized_roundtrip':True,'preserved_source_excluding_student':True,
            'build_sha256':sha(Path(__file__)),'cnet2_pack_sha256':sha(ROOT/'code/cnet2_pack.py'),'packages':proof}
    save(HERE/'packages.json',json.dumps(record,sort_keys=True,indent=2).encode()+b'\n')
    return record


if __name__=='__main__':
    print(json.dumps(build(),ensure_ascii=False),flush=True)
