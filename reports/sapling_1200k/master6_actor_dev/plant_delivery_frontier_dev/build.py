"""Materialize the plant-versus-delivery comparator on the signed Sapling opening carrier."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import zlib

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parent
ROOT=ACTOR.parents[2]
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def sha(b):
    return hashlib.sha256(b).hexdigest()

def build(out=None):
    plan=json.loads((HERE/'protocol.json').read_bytes())
    dep=ACTOR/'endgame_queue_dev'
    op=module(ACTOR/'opening_pair_dev/native.py','opening_front_build')
    packed=op.pack()
    parent=Path(packed).read_bytes()
    assert sha(parent)==plan['parent']
    original=parent.decode().split('def _return_decode(',1)[1].split('"""Action-conditional',1)[0]
    forced='def _front_forced_decode('+original.replace('encoded, apply_residual):','encoded, apply_residual, forced):',1)
    marker='        if not apply_residual:'
    assert forced.count(marker)==1
    forced=forced.replace(marker,"""        if forced[0] == slot:
            if not legal[forced[1]]:
                return None
            action = forced[1]
        if not apply_residual:""")
    queue=module(dep/'build.py','signed_queue_recipe')
    payload=parent+b'\n'
    inputs={p: p.read_bytes() for p in (ROOT/'code/fastsim.py',dep/'policy.py',dep/'build.py',HERE/'runtime.py',HERE/'protocol.json')}
    for p,symbol in ((ROOT/'code/fastsim.py','_QUEUE_SIM_SOURCE'),(dep/'policy.py','_QUEUE_POLICY_SOURCE')):
        payload+=(symbol+' = '+repr(base64.b85encode(zlib.compress(inputs[p],9)))+'\n').encode()
    payload+=queue.RUNTIME.split('_QUEUE_BASE = agent',1)[0].encode()+forced.encode()+inputs[HERE/'runtime.py']
    compile(payload,'candidate.py','exec')
    assert len(payload)<10_000_000
    path=Path(out or "/tmp/sapling-plant-delivery-frontier.py")
    temp=path.with_suffix('.tmp')
    temp.write_bytes(payload)
    temp.replace(path)
    receipt={'pack_sha256':sha(payload),'bytes':len(payload),'parent_sha256':sha(parent),
             'builder_sha256':sha(Path(__file__).read_bytes()),'attached':{str(p.relative_to(ROOT)):sha(v) for p,v in inputs.items()}}
    (HERE/'package.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    return path

if __name__=='__main__':
    print(build())
