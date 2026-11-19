"""Reproducibly attach the two-turn planter decision to the frozen terminal-return learner."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
ROOT=BASE.parents[1]
SOURCE=ROOT/'data/weights/sapling_1200k/terminal_return_fit_v1.py'
FEATURES=BASE/'planter_routes_dev/features.py'
PLAN=HERE/'protocol.json'
WEIGHTS=HERE/'value_weights.json'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def source():
    policy=SOURCE.read_bytes()
    plan=json.loads(PLAN.read_bytes())
    if sha(policy)!=plan['parent_pack_sha256']:
        raise ValueError('economic baseline changed')
    src=policy.decode()
    start=src.rindex('\ndef _mixed_actions(')
    select=src[start:]
    if select.count('return baseline')!=3 or select.count('return candidate')!=1:
        raise ValueError('unrecognized deployed final selection')
    select=select.replace('return baseline','return _OPTION.decide(x,aux,enc,baseline,False)')
    select=select.replace('return candidate','return _OPTION.decide(x,aux,enc,candidate,True)')
    decoder=src[src.index('def _return_decode('):src.index('\n    return labels',src.index('def _return_decode('))+len('\n    return labels')]
    decoder=decoder.replace('_return_decode(x, aux, encoded, apply_residual)','_option_force_decode(x, aux, encoded, apply_residual, force_slot)')
    a,b=decoder.index('        if not apply_residual:'),decoder.index('        labels[slot] = action')
    decoder=decoder[:a]+'''        if slot==force_slot:
            if slot>=10 or x['tok_kind'][slot]!=1 or not legal[0]:
                return None
            action=0
'''+decoder[b:]
    groups=[src[:start],select,decoder,FEATURES.read_text(),'_OPTION_MODEL='+repr(json.loads(WEIGHTS.read_bytes())),
            '_OPTION_CFG='+repr({k:plan[k] for k in ('proposal_window','minimum_first_tenure_net','minimum_remaining_tree_turns','max_price_increase_as_fraction')}),
            (HERE/'runtime.py').read_text()]
    out='\n'.join(groups).encode()
    if len(out)>10_000_000:
        raise ValueError('package size')
    compile(out,'planter_option.py','exec')
    return out

if __name__=='__main__':
    data=source()
    out=HERE/'agent.py'
    if out.exists() and out.read_bytes()!=data:
        raise ValueError('versioned package already exists')
    out.write_bytes(data)
    inputs={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in (SOURCE,FEATURES,PLAN,WEIGHTS,Path(__file__),HERE/'runtime.py')}
    (HERE/'manifest.json').write_text(json.dumps({'pack_sha256':sha(data),'input_sha256':inputs},indent=2)+'\n')
