"""Build a single network with a time-bound allocation of the four already trained worker tensors."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib
import numpy as np

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
EXP=STUDY.parent
KEYS={'U.bias','U.weight','V.bias','V.weight'}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def parse(src):
    names={'_W64','_WMETA','_QMETA'}
    parts={node.targets[0].id:ast.literal_eval(node.value) for node in ast.parse(src).body
           if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in names}
    raw=zlib.decompress(base64.b64decode(parts['_W64']))
    tensors={};off=0
    for key,shape,kind,size in parts['_QMETA']:
        tensors[key]=raw[off:off+size];off+=size
    assert off==len(raw)
    return parts,tensors

def main():
    p=json.loads((HERE/'protocol.json').read_text())
    contract=STUDY/'response/protocol.json'
    assert digest(contract)==p['intervention_contract']
    c=json.loads(contract.read_bytes())['intervention']
    assert [c['first_joint_t'],c['last_joint_t_inclusive']]==p['window']==[128,159]
    refs=p['source']
    parent,whole=EXP/'candidate.py',STUDY/'candidate.py'
    assert (digest(parent),digest(whole))==(refs['parent'],refs['all_steps'])
    pp,wp=EXP/'candidate.npz',STUDY/'candidate.npz'
    assert (digest(pp),digest(wp))==(refs['parent_weights'],refs['trained_weights'])
    with np.load(pp,allow_pickle=False) as a,np.load(wp,allow_pickle=False) as b:
        assert set(a.files)==set(b.files)
        assert {k for k in a.files if not np.array_equal(a[k],b[k])}==KEYS
    source=parent.read_text();other=whole.read_text()
    pm,pw=parse(source);hm,hw=parse(other)
    assert source.replace(pm['_W64'],hm['_W64'])==other
    assert pm['_QMETA']==hm['_QMETA'] and set(k for k in pw if pw[k]!=hw[k])==KEYS
    meta=[m for m in hm['_QMETA'] if m[0] in KEYS]
    raw=b''.join(hw[k] for k,*_ in meta)
    blob=base64.b64encode(zlib.compress(raw,9)).decode()
    init='    if _FEAT is None or observation["step"] == 0 or _FEAT.seat != observation["player"]:\n'
    assert source.count(init)==1
    # Feature history and all autoregressive joint prefixes remain under the original agent.
    body=source.replace(init,'    _select_worker_weights(observation["step"])\n'+init)
    helper="""
def _select_worker_weights(step):
    global _WORKER_BANKS, _WORKER_WINDOW_ACTIVE
    inside = 128 <= step <= 159
    if inside == _WORKER_WINDOW_ACTIVE:
        return
    weights = _NETS[1].w
    if _WORKER_BANKS is None:
        trained = _dequantize(zlib.decompress(base64.b64decode(_WINDOW_BYTES)), _WINDOW_META)
        for name in ("U.weight", "V.weight"):
            trained[name + "T"] = np.ascontiguousarray(trained[name].T)
        original = {name: weights[name] for name in trained}
        _WORKER_BANKS = (original, trained)
    weights.update(_WORKER_BANKS[int(inside)])
    _WORKER_WINDOW_ACTIVE = inside
"""
    body+=f'\n_WINDOW_BYTES = {blob!r}\n_WINDOW_META = {meta!r}\n_WORKER_BANKS = None\n_WORKER_WINDOW_ACTIVE = False\n'+helper
    out=HERE/'candidate.py';out.write_text(body)
    assert out.stat().st_size<10_000_000
    log=dict(candidate=digest(out),bytes=out.stat().st_size,extra_file_bytes=out.stat().st_size-parent.stat().st_size,
             included_tensor_bytes=len(raw),window_quantized_sha256=hashlib.sha256(raw).hexdigest(),
             embedded_subset=meta,all_tensor_provenance=refs,protocol=digest(HERE/'protocol.json'),
             parent_entire_source_restored=body[:body.index('\n_WINDOW_BYTES =')].replace('    _select_worker_weights(observation["step"])\n','')==source)
    assert log['parent_entire_source_restored']
    (HERE/'pack.json').write_text(json.dumps(log,indent=2)+'\n')

if __name__=='__main__':main()
