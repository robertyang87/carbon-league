"""Attach the private/public boundary and force exactly one legal base label in a fresh frozen-v1 decode."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[3]
SRC=BASE/'data/weights/sapling_1200k/terminal_return_fit_v1.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def rebuild():
    reg=json.loads((HERE/'protocol.json').read_bytes())
    if sha(SRC)!=reg['parent_sha256']:raise ValueError('unexpected source policy')
    src=SRC.read_text();start=src.index('def _return_decode(');end=src.index('\n    return labels',start)+len('\n    return labels')
    d=src[start:end].replace('def _return_decode(x, aux, encoded, apply_residual):','def _lw_redecode(x, aux, encoded, apply_residual, focal):')
    lo=d.index('        if not apply_residual:');hi=d.index('        labels[slot] = action',lo)
    d=d[:lo]+'''        if slot==focal:
            if not legal[1]:return None
            action=1
'''+d[hi:]
    pos=src.rindex('\ndef _mixed_actions(');tail=src[pos:]
    assert tail.count('return baseline')==3 and tail.count('return candidate')==1
    tail=tail.replace('return baseline','return _lw_select(x,aux,enc,baseline,False)').replace('return candidate','return _lw_select(x,aux,enc,candidate,True)')
    code='\n'.join((src[:pos],tail,(HERE/'policy.py').read_text(),d,(HERE/'wrapper.py').read_text()))
    data=code.encode();compile(data,'late_recruit_agent.py','exec')
    if len(data)>10_000_000:raise ValueError('candidate too large')
    out=HERE/'agent.py'
    if out.exists() and out.read_bytes()!=data:raise ValueError('released source changed')
    out.write_bytes(data)
    fields={'policy_sha256':sha(out),'parent_sha256':sha(SRC),'bytes':len(data),'inputs':{name:sha(HERE/name) for name in ('protocol.json','build.py','policy.py','wrapper.py')}}
    (HERE/'manifest.json').write_text(json.dumps(fields,indent=2)+'\n')
    return out


if __name__=='__main__':print(rebuild())
