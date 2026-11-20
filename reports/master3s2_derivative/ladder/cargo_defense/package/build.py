"""Build a stand-alone heavy-collector defense from the signed parental bytes and public risk head."""
import ast
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
PARENT=HERE.parents[2]/'cold_start/candidate.py'
WEIGHT_SHA='988401e2d290355c971eab06a1c213a55f271579c9a979411689c0164c6bd8fe'
PARENT_SHA='7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def build():
    base=PARENT.read_bytes()
    frozen=(STUDY/'weights.json').read_bytes()
    w=json.loads(frozen)
    run=(STUDY/'runtime.py').read_text()
    held=json.loads((STUDY/'held.json').read_bytes())
    assert sha(base)==PARENT_SHA and sha(frozen)==WEIGHT_SHA==held['weight_sha256'] and held['verdict']=='PASS'
    assert w['protocol']==sha((STUDY/'protocol.json').read_bytes()) and len(w['weights'])==22
    assert 0<=w['recruit_probability']<.25
    definitions=[n for n in ast.parse(run).body if isinstance(n,ast.FunctionDef)]
    names={n.name:'_cargo_'+n.name for n in definitions}
    names.update(DEST='_OPP_STEPS',DIST='FEAT_DIST')
    bare='\n\n'.join(ast.get_source_segment(run,n) for n in definitions)
    embedded=re.sub(r'\b(?:'+'|'.join(names)+r')\b',lambda k:names[k[0]],bare)
    header='_CARGO_W = np.array('+repr(w['weights'])+',dtype=np.float32)\n_CARGO_RECRUIT = '+repr(w['recruit_probability'])+'\n\n'
    addition=(header+embedded+'\n\n'+(HERE/'insert.py').read_text()+'\n\n').replace('\n','\r\n').encode()
    probe=b'def _search(obs, configuration, est, act0, lw0, lb0, prev, t0, home=()):'
    marker=b"    st['tr'].set_actions(players, act)\r\n"
    insert=b"""    if configuration is not None and _CARGO_DEFENSE:\r
        act = _cargo_postsearch(obs, st['prev'], act, t0)\r
"""
    assert base.count(probe)==base.count(marker)==1 and b'_CARGO_DEFENSE' not in base
    candidate=base.replace(probe,addition+probe).replace(marker,insert+marker)
    assert candidate.replace(addition,b'').replace(insert,b'')==base
    assert len(candidate)<=10_000_000
    compile(candidate,'candidate.py','exec')
    return candidate,dict(parent_sha256=sha(base),candidate_sha256=sha(candidate),weight_sha256=sha(frozen),
                          runtime_sha256=sha(run.encode()),insert_sha256=sha((HERE/'insert.py').read_bytes()),build_sha256=sha(Path(__file__).read_bytes()),
                          bytes=len(candidate),cutoffs=dict(start_reserve_ms=12,worker_reserve_ms=6,return_reserve_ms=2),
                          action_order='After native search and before belief tracker records our issued joint movement',
                          model=dict(coefficients=22,recruit_probability=w['recruit_probability']))


if __name__=='__main__':
    cand,meta=build()
    for path,raw in [(HERE/'candidate.py',cand),(HERE/'manifest.json',(json.dumps(meta,indent=2,sort_keys=True)+'\n').encode())]:
        if path.exists() and path.read_bytes()!=raw:
            raise ValueError("Changed frozen artifact "+str(path))
        path.write_bytes(raw)
    print(meta['candidate_sha256'],meta['bytes'])
