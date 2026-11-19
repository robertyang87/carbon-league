"""Freeze a bounded additional training intake before the previous registered queue finishes."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

OUT=Path(__file__).resolve().parent
TRAIN=OUT.parent
sys.path.insert(0,str(TRAIN))
from source_games import PLAN,ROOT,gate
from select_roots import load,retain,sha

NEXT_MAP_PAIRS=((621476120,621476121),(621477120,621477121),(621478120,621478121))

def make():
    original=load(TRAIN/'next_intake.json')
    witness=TRAIN/'batches/intake_03/aggregate.json'
    evidence=load(witness)
    primary=[s for b in original['batches'] for s in b['maps']]
    protected=original['protected_unopened_development_maps']
    already={*PLAN['train_maps'],*primary,*protected}
    locked={**load(TRAIN/'clearance.json')['boards'],**{s:h for b in original['batches'] for s,h in b['terrain_at_queue_time'].items()}}
    flat=[s for p in NEXT_MAP_PAIRS for s in p]
    ladder,ladder_seeds,ladder_boards=gate.map_identity.ladder_boards()
    if len(set(flat))!=6 or set(flat)&already or evidence['filled_pairs']+sum(b['max_pairs'] for b in original['batches'][-2:])>=32:
        raise ValueError('future maps overlap or registration is not a necessary new data block')
    eligible=gate.map_identity.checked_maps(flat,set(ladder_seeds)|already,set(ladder_boards)|set(locked.values()))
    identities={row['seed']:row['initial_terrain_u8_sha256'] for row in eligible}
    groups=[]
    for group,maps in enumerate(NEXT_MAP_PAIRS,6):
        jobs=[]
        for k,s in enumerate(maps):
            idx=2*(group-1)+k
            for seat in (0,1):
                source_id=f"{PLAN['parent_pack_sha256']}/{s}/{seat}"
                for window in range(len(PLAN['windows'])):
                    bits=[seat^window^(idx&1),seat^((idx>>1)&1),window^((idx>>2)&1)]
                    jobs.append({'case_id':source_id+'/'+str(window),'source_id':source_id,'seat':seat,'seed':s,'window':window,
                                 'global_map_index':idx,'rng_seed':813700+4*idx+2*seat+window,'desired_exposure_bits':bits})
        groups.append({'name':f'intake_{group:02d}','maps':list(maps),'terrain_at_queue_time':{str(s):identities[s] for s in maps},
                       'activation':'fill_only_if_fewer_than_32_realized_purchases_or_8_training_maps',
                       'pending_full_historical_clearance':True,'expected_source_games':4,'max_pairs':8,'jobs':jobs})
    old_jobs=[j for b in original['batches'] for j in b['jobs']]
    new_jobs=[j for b in groups for j in b['jobs']]
    if len({j['case_id'] for j in old_jobs+new_jobs})!=len(old_jobs+new_jobs) or len({j['rng_seed'] for j in old_jobs+new_jobs})!=len(old_jobs+new_jobs):
        raise ValueError('overlap with previously registered cases or original opponent RNG')
    previous_03=load(TRAIN/'batches/intake_03/protocol.json')
    return {'schema':'sapling-tree-training-supplement-v1','state':'frozen_before_previous_intake_05_completion',
            'registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'revision_at_registration':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'register_program_sha256':sha(Path(__file__)),'original_protocol_sha256':sha(TRAIN/'protocol.json'),
            'original_queue_sha256':sha(TRAIN/'next_intake.json'),'original_03_aggregate_sha256':sha(witness),
            'original_05_must_seal':True,
            'expected_original_05_prefix':['.']+['batches/'+b['name'] for b in original['batches']],
            'policy_sha256':PLAN['parent_pack_sha256'],'weight_manifest_sha256':PLAN['parent_weight_manifest_sha256'],
            'student_npz_sha256':PLAN['student_npz_sha256'],'examiner_md5':PLAN['examiner_md5'],
            'selector_at_registration_sha256':previous_03['selector_at_registration_sha256'],
            'shared_decode_at_registration_sha256':previous_03['shared_decode_at_registration_sha256'],
            'frozen_ladder_at_registration':ladder,'protected_unopened_development_maps':protected,
            'previous_predeclared_terrain':locked,'goals':original['goals'],
            'justification':{'maps_verified':evidence['distinct_maps'],'root_pairs_verified':evidence['complete_pairs'],
                             'actual_payments_verified':evidence['filled_pairs'],'maximum_actual_payments_after_05':evidence['filled_pairs']+16},
            'procedure':'Seal and validate all registered original intakes through 05 first. Check only actual completed purchase count and distinct map count before each additional whole block. Stop at 32 purchases and 8 maps, regardless of observed effect sign; evaluate fit sign support separately. Reserve all unused and failed source IDs. One owner per intake, no repeated cases, no formal/development map starts; only maps that pass fresh seed/range/terrain clearance may run. Keep all official scores and every empty window or unsuccessful payment. After the third inactive/used tranche there is no implicit refill. Cross-map fit and independent development retain the original rules.',
            'batches':groups}

if __name__=='__main__':
    path=OUT/'registration.json'
    if not path.exists():
        if (TRAIN/'batches/intake_05/aggregate.json').exists():raise RuntimeError('must preregister before old 05 result')
        retain(path,make())
    q=load(path)
    if q['register_program_sha256']!=sha(Path(__file__)) or q['original_queue_sha256']!=sha(TRAIN/'next_intake.json'):
        raise ValueError('registration archive changed')
    print(sha(path),q['justification'],flush=True)
