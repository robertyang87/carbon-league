"""Verify the closed experimental evidence package without fetching data or starting a game."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from verify_sources import verify

HERE=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    verify()
    registry=json.loads((HERE/'sources_check.json').read_text())
    assert registry['native_trajectories']==54 and registry['unique_physical_seeds']==41
    audit=json.loads((HERE/'input_audit.json').read_text())
    assert sum(i['carriers'] for i in audit['cases'])==67 and audit['invalid_histories_rejected']==72
    score=json.loads((HERE/'evaluation.json').read_text())
    assert score['weights_sha256']==digest(HERE/'head.npz')
    for k,m,n in [('train',17,42333),('calibration',4,9782),('native_all_held',6,14121),('ladder',14,29803)]:
        p=score['panels'][k];assert (p['maps'],p['n'])==(m,n)
    for panel in ('native','ladder'):
        with np.load(HERE/(panel+'_features.npz'), allow_pickle=False) as z:
            assert len(z['action'])==sum(score['panels'][k]['n'] for k in (['train','calibration','native_all_held'] if panel=='native' else ['ladder']))
            assert np.isfinite(z['x']).all() and ((0<=z['action'])&(z['action']<5)).all()
    with gzip.open(HERE/'commercial.json.gz','rt') as f:comm=json.load(f)
    for key,name in [('head_sha256','head.npz'),('model_sha256','model.py'),('sources_sha256','sources.json'),('code_sha256','commercial.py'),('arms_extractor_sha256','commercial_arms.py')]:
        assert comm[key]==digest(HERE/name)
    rows=comm['rows'];hold=[r for r in rows if r['split']=='held']
    assert len(rows)==108 and len(hold)==28 and all(r['policies']['forecast']['selected']==r['policies']['uniform']['selected'] for r in hold)
    assert all(r['outcomes']['forecast']==r['outcomes']['uniform'] for r in hold)
    assert comm['reports']['held']['forecast']['changes']==2 and comm['reports']['all']['forecast']['equal_map_delta']['own_normal_deposit']<0
    with gzip.open(HERE/'collision_audit.json.gz','rt') as f:coll=json.load(f)
    native=coll['summary']['native_held'];league=coll['summary']['ladder']
    assert (native['distinct_victim_transfers'],native['empty_hunter_visibly'])==(7,5) and (league['distinct_victim_transfers'],league['empty_hunter_visibly'])==(35,26)
    names=[p for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='verification.json']
    out={'sha256':{p.name:digest(p) for p in names},'file_count_excluding_receipt':len(names),'total_bytes_excluding_receipt':sum(p.stat().st_size for p in names),
         'source_registry_sha256':registry['native_registry_sha256'],'distinct_map_seeds':41,'commercial_holdout_actions_identical_to_uniform':True,
         'commercial_held_n':28,'new_native_games_launched':0,'independent_exam_opened':False}
    (HERE/'verification.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':run()
