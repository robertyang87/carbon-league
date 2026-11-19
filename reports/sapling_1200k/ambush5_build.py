"""Build the public-information ambush candidate on the frozen V5 actor."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

from ambush5_learned import fit, signed, frame, vector, project, context
from ambush5_probe import pairs, approach, choose, HERE


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replays',type=Path,required=True)
    args=p.parse_args()
    root=HERE.parents[1]
    parent=root/'data/weights/sapling_1200k/soft_it50_v5_agent.py'
    raw=parent.read_bytes()
    if sha(raw)!='dceed7891e9ff25cf9e211e1c9b61d979d27a253b4c0935f2439804372b31389':
        raise ValueError('frozen V5 parent changed')
    model,split=fit(args.replays)
    model.min_probability=.15
    base={d:model.offsets(d,None,None) for d in (3,4,5)}
    coarse={(d,c):model.offsets(d,c,None) for d,c in model.hist[1]}
    detailed={(d,f):model.offsets(d,f[0],f) for d,f in model.hist[2]}
    # Only store overrides. All unseen detailed contexts fall back to their observed coarse context.
    detailed={k:v for k,v in detailed.items() if v!=coarse.get((k[0],k[1][0]),base[k[0]])}
    coarse={k:v for k,v in coarse.items() if v!=base[k[0]]}
    prefix="""
from collections import Counter
_geom = sys.modules['cnet2_feat']
DIST, DEST, MOVES = _geom.DIST.tolist(), _geom.DEST.tolist(), tuple(_geom.MOVES)
intent = types.SimpleNamespace(MOVE_ID={name:n for n,name in enumerate(MOVES)})
"""
    funcs='\n\n'.join(inspect.getsource(f) for f in (signed,frame,vector,project,context,pairs,approach))
    selection=inspect.getsource(choose)
    a=selection.index('        if uid not in prediction:')
    b=selection.index('        original =',a)
    selection=selection[:a]+"""        if uid not in prediction:
            prediction[uid] = forecaster.crossings(obs,previous,uid)
"""+selection[b:]
    selection=selection.replace('own_orders, future=None, forecaster=None','own_orders, forecaster').replace('Only the oracle variant is passed realized future states.','Select a crossing using the public-board forecast.')
    tables=f"_BASE={base!r}\n_COARSE={coarse!r}\n_DETAILED={detailed!r}\n"
    body=prefix+tables+funcs+'\n\n'+selection+'\n\n'+(HERE/'ambush5_runtime.py').read_text()
    if 'future' in body:
        raise ValueError('unexpected lookahead field in runtime')
    out=parent.with_name('soft_it50_v5_ambush5.py')
    data=raw+b'\n'+body.encode()
    compile(data,str(out),'exec')
    if len(data)>10_000_000 or (out.exists() and out.read_bytes()!=data):
        raise ValueError('candidate immutable or oversized')
    if not out.exists():
        out.write_bytes(data)
    receipt={'pack':str(out.relative_to(root)),'sha256':sha(data),'bytes':len(data),'parent_sha256':sha(raw),
             'parent_weights_sha256':sha(parent.with_name('soft_it50_v5.npz').read_bytes()),
             'training_maps':split['train'],'training_roots':sum(model.hist[0][3,()].values()),
             'probability_min':.15,'expected_cargo_min':25,'table_entries':{'coarse':len(coarse),'detailed':len(detailed)}}
    (HERE/'ambush5_build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt['sha256'],receipt['bytes'])


if __name__=='__main__':
    main()
