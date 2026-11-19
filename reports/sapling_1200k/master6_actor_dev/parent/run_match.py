"""Run either parent or the separately frozen same-wrapper cash_v3 student on the registered CPU pair."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT/'code'))
import cnet2_pack

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def wrapper(pack):
    # Keep every expression, policy option and all module source. The three
    # packed weight fields contain quantized tensors and training provenance.
    m=ast.parse(pack.read_bytes())
    shapes=None
    for n in m.body:
        if not isinstance(n,ast.Assign) or len(n.targets)!=1 or not isinstance(n.targets[0],ast.Name):continue
        if n.targets[0].id not in ('_QMETA','_WMETA','_W64'):continue
        if n.targets[0].id=='_QMETA':
            shapes=[x[:3] for x in ast.literal_eval(n.value)]
        n.value=ast.Constant('weight-payload')
    if not shapes:raise ValueError('missing int8 entries')
    return hashlib.sha256(ast.dump(m).encode()).hexdigest(), shapes

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('arm',choices=['parent','candidate'])
    p.add_argument('--pack',required=True,type=Path);p.add_argument('--weights',required=True,type=Path)
    p.add_argument('--pack-sha256',required=True);p.add_argument('--weights-sha256',required=True)
    a=p.parse_args();pack=a.pack.resolve();w=a.weights.resolve();out=HERE if a.arm=='parent' else HERE.parent/'candidate'
    parent=ROOT/'data/weights/sapling_1200k/cash_v3_agent.py';reg=json.loads((HERE/'registration.json').read_bytes())
    assert reg['seed']==621458920 and reg['cpus']==[1,2] and reg['frozen_parent_sha256']==sha(parent)
    assert sha(pack)==a.pack_sha256 and sha(w)==a.weights_sha256 and wrapper(pack)==wrapper(parent)
    if a.arm=='candidate' and sha(pack)==sha(parent):raise ValueError('candidate equals parent')
    # Reconstitute the released pack directly from this arm's linked real NPZ.
    with tempfile.TemporaryDirectory() as d:
        scratch=Path(d)/'weights.npz';scratch.write_bytes(w.read_bytes())
        target=Path(d)/'rebuilt.py';cnet2_pack.QMODE='int8';cnet2_pack.main(str(scratch),str(target),teacher=2)
        if sha(target)!=sha(pack):raise ValueError('cannot rebuild frozen artifact from supplied weights')
    cpus=['1','2'] if a.arm=='parent' else ['0','3']
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    subprocess.run([sys.executable,'-u',str(BASE/'tree_pack_arena.py'),'--maps',str(reg['seed']),'--cpus',*cpus,
                    '--weights',str(w),'--pack',str(pack),'--pack-sha256',sha(pack),'--run',str(out)],env=env,check=True)
    proof=out/'official_replay.json'
    if not proof.exists():
        subprocess.run([sys.executable,str(BASE/'official_game_replay.py'),str(out),'--out',str(proof)],env=env,check=True)
    report=json.loads((out/'results.json').read_bytes());r=json.loads(proof.read_bytes())['runs'][out.name]['games']
    assert report['status']=='complete' and len(r)==2 and {x['seat'] for x in r}=={0,1}
    for row,official in zip(report['games'],r):
        assert sha(out/row['raw_file'])==official['transcript_sha256']==row['raw_sha256'] and row['valid'] and not row['errors']
        assert (official['map_sha256']==reg['initial_terrain_u8_sha256']==row['initial_terrain_u8_sha256']
                and row['official_score_by_seat']==official['official_reward_by_seat'] and official['statuses']==['DONE','DONE'])
if __name__=='__main__':main()
