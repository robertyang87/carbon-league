"""Document historical and new training registrations against the two sealed intervals and contingent new board."""
import gzip,json,tempfile
from pathlib import Path
import sys
H=Path(__file__).resolve().parent;T=H.parent;A=T.parent
sys.path.insert(0,str(A/'first_win_dev'))
import native

plan=native.read(H/'protocol.json');panel=native.read(T/'panel.json')['games']
prev=native.read(A/'first_win_dev/clearance.json')['physical_boards']
index={g['terrain_sha256']:g['seed'] for g in panel}
with tempfile.TemporaryDirectory() as t:
    native.HERE=Path(t);native.PLAN=H/'protocol.json';native.REG=native.sha(native.PLAN)
    clearance=native.clearance()
fresh=clearance['physical_boards']['621758339'];excluded=[]
for c in plan['exposures']:
    assert prev[str(c['seed'])]==c['terrain'] and c['terrain'] not in index and c['seed'] not in index.values()
    excluded.append(dict(seed=c['seed'],terrain=c['terrain'],first_use='previously verified, original-examiner exposure',final_training_and_holdout_overlaps=0))
assert fresh not in index and 621758339 not in index.values()
old=set()
for f in (A/'first_win_dev/clearance.json', A/'elite_role_mix_dev/conditional/fresh_paired_dev/clearance.json'):
    old.update(native.read(f)['physical_boards'].values())
assert fresh not in old
native.save(H/'split_clearance.json',dict(protocol_sha256=native.REG,training_panel_sha256=plan['panel_sha256'],frozen_exposures=excluded,present_training_and_holdout_maps=len(panel),fresh_held_in_reserve=dict(seed=621758339,terrain=fresh,existing_terrain_overlaps=0,existing_seed_overlaps=0),old_registered_seeds=clearance['old_unique_seeds'],old_registered_terrains=clearance['old_unique_terrains'],original_manifest_sha256=native.sha(A/'first_win_dev/clearance.json')))
