"""Compose the public route option with the frozen double-network actor and its real Prefix."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
ROOT=HERE.parents[3]
PARENT=ROOT/'data/weights/sapling_1200k/terminal_return_fit_v1.py'
MODEL=HERE.parent/'planter_option/value_weights.json'
FEATURES=BASE/'planter_routes_dev/features.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def rebuild():
    p=json.loads((HERE/'protocol.json').read_bytes())
    if sha(PARENT)!=p['parent'] or sha(MODEL)!=p['value_sha256']:
        raise ValueError('parent or conditional value model differs')
    src=PARENT.read_text()
    cut=src.rindex('\ndef _mixed_actions(')
    select=src[cut:]
    if select.count('return baseline')!=3 or select.count('return candidate')!=1:
        raise ValueError('unrecognized v1 safety boundaries')
    select=select.replace('return baseline','return _APP.decide(x,aux,enc,baseline,False)').replace('return candidate','return _APP.decide(x,aux,enc,candidate,True)')
    lo=src.index('def _return_decode(');hi=src.index('\n    return labels',lo)+len('\n    return labels')
    decoder=src[lo:hi].replace('_return_decode(x, aux, encoded, apply_residual)','_approach_decode(x, aux, encoded, apply_residual, focal, order)')
    lo=decoder.index('        if not apply_residual:');hi=decoder.index('        labels[slot] = action',lo)
    decoder=decoder[:lo]+'''        if slot==focal:
            if slot>=10 or x["tok_kind"][slot]!=1 or not legal[order] or combined[action]-combined[order]>1.5:
                return None
            action=order
'''+decoder[hi:]
    data='\n'.join((src[:cut],select,decoder,FEATURES.read_text(),'_APP_MODEL='+repr(json.loads(MODEL.read_bytes())),(HERE/'runtime.py').read_text())).encode()
    compile(data,'approach_agent.py','exec')
    manifest={'pack_sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'inputs':{str(t.relative_to(ROOT)):sha(t) for t in (PARENT,MODEL,FEATURES,HERE/'protocol.json',HERE/'build.py',HERE/'runtime.py')}}
    if len(data)>10_000_000:raise ValueError('oversized candidate')
    return data,manifest

def write():
    data,manifest=rebuild()
    out=HERE/'agent.py'
    if out.exists() and out.read_bytes()!=data:raise ValueError('pack changed since release')
    out.write_bytes(data)
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':write()
