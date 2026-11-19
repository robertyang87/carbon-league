"""Build the frozen Sapling V5 collector interception candidate."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
WEIGHTS=ROOT/'data/weights/sapling_1200k'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parent=WEIGHTS/'soft_it50_v5_agent.py'
    raw=parent.read_bytes()
    if sha(raw)!='dceed7891e9ff25cf9e211e1c9b61d979d27a253b4c0935f2439804372b31389':
        raise ValueError('unexpected actor')
    head,marker,_=raw.decode().partition('\n_AG = None\n')
    if not marker:raise ValueError('missing standard policy loader')
    fitted=json.loads((HERE/'interception_fit.json').read_bytes())
    intent=ROOT/'reports/master_family/tactical_rules/opponent_intent.py'
    if sha(intent.read_bytes())!=fitted['source_sha256']['opponent_intent.py']:
        raise ValueError('intent definition has changed')
    source=intent.read_text()
    defs={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    names=['features','probabilities','predict']
    prelude='from cnet2_feat import DEST, DIST, MOVES, move_between\n'
    prelude+='FEATURES = '+repr(fitted['features'])+'\n_INTENT_WEIGHTS = np.array('+repr(fitted['weights'])+',dtype=float)\n'
    body='\n\n'.join(defs[k] for k in names)
    layer=HERE/'interception_layer.py'; runtime=HERE/'interception_runtime.py'
    result=(head+'\n'+prelude+'\n'+body+'\n'+layer.read_text()+'\n'+runtime.read_text()).encode()
    out=WEIGHTS/'soft_it50_v5_interception.py'
    out.write_bytes(result)
    receipt=dict(candidate='one frozen int8 V5 network + Top Five move-probability interception',
                 candidate_sha256=sha(result),bytes=len(result),net_parameters=1206476,intent_parameters=32,
                 source_sha256={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in
                                (parent,WEIGHTS/'soft_it50_v5.npz',intent,HERE/'interception_fit.json',
                                 layer,runtime,Path(__file__))})
    (HERE/'interception_build.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    main()
