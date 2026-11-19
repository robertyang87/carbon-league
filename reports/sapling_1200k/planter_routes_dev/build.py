"""Bind fitted route valuation and source files to the frozen Sapling+it50 package."""
from pathlib import Path
import hashlib
import json
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from build_it50_endhome import PARENT,PARENT_SHA
OUT=PARENT.with_name('planter_routes_it50_v3.py')

def digest(b):return hashlib.sha256(b).hexdigest()
def build():
    original=PARENT.read_bytes()
    if digest(original)!=PARENT_SHA:raise ValueError('parent identity')
    patches=[('def _mixed_actions(x, aux):','def _route_encode(x):'),
             ('    prefix = F.Prefix(x, aux)',"    return states\n\n\ndef _route_decode(x, aux, encoded, changes=None):\n    F = sys.modules['cnet2_feat']\n    states = [list(state) for state in encoded]\n    prefix = F.Prefix(x, aux)"),
             ('        labels[slot] = action','        if changes is not None:\n            action = _route_decide(F,x,prefix,slot,action,combined,legal,changes)\n        labels[slot] = action')]
    src=original.decode()
    for old,new in patches:
        if src.count(old)!=1:raise ValueError('decoder site')
        src=src.replace(old,new)
    parent_runtime=(HERE.parent/'it50_endhome_runtime.py').read_text()
    src+='\n'+parent_runtime[parent_runtime.index('def _return_destinations'):parent_runtime.index('def _mixed_actions')]
    src+='\n'+(HERE/'features.py').read_text()+'\n_ROUTE_MODEL = '+repr(json.loads((HERE/'weights.json').read_bytes()))+'\n'+(HERE/'runtime.py').read_text()
    compile(src,str(OUT),'exec');data=src.encode()
    if len(data)>10_000_000 or OUT.exists() and OUT.read_bytes()!=data:raise ValueError('frozen output or size limit')
    OUT.write_bytes(data)
    meta={p:digest((HERE/p).read_bytes()) for p in ['plan.json','split.json','weights.json','features.py','runtime.py','train.py','build.py','../it50_endhome_runtime.py']}
    meta.update(parent_sha256=PARENT_SHA,candidate_sha256=digest(data),package_bytes=len(data))
    (HERE/'build_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(digest(data))
if __name__=='__main__':build()
