"""Verify the archived 54-tape world without reading the rolling main-tip registry."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
REF='3d02c08f78e4a2efe954d882c3b8e8648aed2411'
REGISTRY='reports/sapling_1200k/master6_actor_dev/v1_phase_census/latest_games.json'


def verify():
    v=json.loads((HERE/'sources.json').read_text())
    raw=subprocess.check_output(['git','show',f'{REF}:{REGISTRY}'],cwd=ROOT)
    assert hashlib.sha256(raw).hexdigest()==v['native_source_sha256']
    cols={(r['seed'],r['seat'],r['terrain'],r['source'],r['sha256']) for r in v['native']}
    old=json.loads(raw)['games']
    assert cols=={(r['seed'],r['seat'],r['map'],r['sources'][0],r['raw_sha256']) for r in old} and len(cols)==54
    both=v['native']+v['ladder'];assert len({(r['id'],r['seat']) for r in both})==len(both)
    sets={k:{r['terrain'] for r in both if r['split']==k} for k in sorted({r['split'] for r in both})}
    for i,(l,a) in enumerate(sets.items()):
        for name,b in list(sets.items())[i+1:]:assert not a.intersection(b)
    seed_terrain={}
    for r in both:
        assert seed_terrain.setdefault(r['seed'],r['terrain'])==r['terrain']
    if len(seed_terrain)!=sum(map(len,sets.values())):
        raise ValueError('a distinct seed maps to a previously used terrain')
    for r in v['native']:assert hashlib.sha256((ROOT/r['source']).read_bytes()).hexdigest()==r['sha256']
    out=dict(native_registry_revision=REF,native_registry_file=REGISTRY,native_registry_sha256=v['native_source_sha256'],
             native_trajectories=len(old),ladder_games=len(sets['ladder']),map_counts={k:len(a) for k,a in sets.items()},
             native_registered_opponents=['original Master3s2'],ladder_registered_names=sorted({r['opponent'] for r in v['ladder']}),
             overlap_physical_terrains=0,unique_physical_seeds=len(seed_terrain),source_manifest_sha256=hashlib.sha256((HERE/'sources.json').read_bytes()).hexdigest())
    return out


if __name__=='__main__':
    r=verify();(HERE/'sources_check.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(r['map_counts'])
