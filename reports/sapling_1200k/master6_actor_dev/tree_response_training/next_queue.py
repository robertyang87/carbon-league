"""Build a read-only continuation ledger. This command never schedules or plays a game."""
import argparse
import json
from pathlib import Path
from source_games import HERE,PLAN,ROOT,gate
from select_roots import load,retain,sha

# Distinct from the two held-out maps. Candidate seeds require fresh clearance when a future intake is authorized.
NEXT_MAP_PAIRS=((621472120,621472121),(621473120,621473121),(621474120,621474121),(621475120,621475121))
def build():
    initial=load(HERE/'training_labels.json')
    filled=[r for r in initial['pairs'] if r['pay_intent_filled']]
    signs={'positive':sorted({r['seed'] for r in filled if r['target_delta_official_margin']>0}),
           'negative':sorted({r['seed'] for r in filled if r['target_delta_official_margin']<0})}
    activated=all(signs.values())
    old_terrains={**load(HERE/'clearance.json')['boards']}
    ladder,observed_seeds,observed_boards=gate.map_identity.ladder_boards()
    flattened=[x for pair in NEXT_MAP_PAIRS for x in pair]
    if len(set(flattened+PLAN['train_maps']+PLAN['development_maps']))!=len(flattened)+4:
        raise ValueError('overlap in continuation schedule')
    boards=gate.map_identity.checked_maps(flattened,observed_seeds|set(map(int,old_terrains)),observed_boards|set(old_terrains.values()))
    by_seed={r['seed']:r['initial_terrain_u8_sha256'] for r in boards}
    queue=[]
    if activated:
        for i,maps in enumerate(NEXT_MAP_PAIRS):
            batch=f'intake_{i+2:02d}'
            tasks=[]
            for k,s in enumerate(maps):
                idx=2+2*i+k
                for seat in PLAN['physical_seats']:
                    source_id=f'{PLAN["parent_pack_sha256"]}/{s}/{seat}'
                    for w in range(len(PLAN['windows'])):
                        bits=[seat^w^(idx&1),seat^((idx>>1)&1),w^((idx>>2)&1)]
                        tasks.append({'case_id':source_id+'/'+str(w),'source_id':source_id,'seat':seat,'seed':s,'window':w,
                                      'global_map_index':idx,'rng_seed':813700+4*idx+2*seat+w,'desired_exposure_bits':bits})
            queue.append({'name':batch,'maps':list(maps),'terrain_at_queue_time':{str(s):by_seed[s] for s in maps},
                          'activation':'primary' if i<3 else 'fill_only_if_fewer_than_32_realized_purchases_or_8_training_maps',
                          'pending_full_historical_clearance':True,'expected_source_games':4,'max_pairs':8,'jobs':tasks})
    return {'state':'registered_for_later_not_started' if activated else 'no_continuation_due_to_one_sided_or_unfilled_labels',
            'source_data_sha256':sha(HERE/'training_labels.json'),'policy_sha256':PLAN['parent_pack_sha256'],
            'protected_unopened_development_maps':PLAN['development_maps'],'ladder_fingerprint_at_queue_time':ladder,
            'already_verified':{'independent_source_maps':initial['unique_source_maps'],'completed_pairs':len(initial['pairs']),
                                'actually_paid_with_dual_official_terminal':len(filled),'maps_per_realized_effect_sign':signs},
            'goals':{'independent_training_maps':8,'completed_actually_paid_pairs':32,'minimum_distinct_maps_per_sign':2},
            'remaining_minimum':{'new_training_maps':max(0,8-initial['unique_source_maps']),
                                 'new_actually_paid_pairs':max(0,32-len(filled))},
            'procedure':'A separate future intake owner verifies clearance against the then-current ledger, freezes local protocol, and reuses all already hash-verified source and response receipts. No commands in this script execute games. Do not resample completed cases, map/seat/source, unfavorable labels or genuine no-payment. Do not overwrite this first intake or touch protected development maps. After the six primary maps evaluate counts only; use the pre-listed fill-only pair if 32 actual payments or eight independent maps are still missing, regardless of the observed reward signs. Do not start that pair once counts are reached. If the fill block also exhausts, a further independent block must receive separate registration before play. Rotate the requested three exposure bits so high enemy freight does not remain identical to source-map identity; retain exact and fallback strata.',
            'batches':queue}

def next_phase(saved):
    if not saved['batches']:
        return None
    valid=saved['already_verified']['actually_paid_with_dual_official_terminal']
    nmap=saved['already_verified']['independent_source_maps']
    for batch in saved['batches']:
        if batch['activation']!='primary' and valid>=32 and nmap>=8:
            continue
        base=HERE/'batches'/batch['name']
        if not (base/'clearance.json').is_file() or not (base/'protocol.json').is_file():
            return [batch['name'],'check_map_and_freeze_protocol']
        clearance=load(base/'clearance.json')
        protocol=load(base/'protocol.json')
        if clearance.get('verdict')!='eligible_in_checked_scope' or protocol.get('train_maps')!=batch['maps'] or protocol.get('parent_pack_sha256')!=saved['policy_sha256']:
            return [batch['name'],'reconcile_stale_clearance_or_protocol_before_scheduling']
        required={(s,k) for s in batch['maps'] for k in (0,1)}
        if not (base/'source_receipt.json').is_file():
            return [batch['name'],'collect_or_verify_missing_source_map_seats']
        games=load(base/'source_receipt.json').get('games',[])
        if {(x['seed'],x['seat']) for x in games}!=required or len(games)!=4:
            return [batch['name'],'verify_incomplete_or_duplicate_source_receipts']
        if not (base/'opportunities.jsonl.gz').is_file():
            return [batch['name'],'export_all_public_opportunities']
        if not (base/'selection.json').is_file():
            return [batch['name'],'freeze_public_root_set']
        if not (base/'training_labels.json').is_file():
            return [batch['name'],'verify_and_resume_only_missing_paired_branches']
        d=load(base/'training_labels.json')
        if d.get('selection_sha256')!=sha(base/'selection.json') or d.get('protocol_sha256')!=sha(base/'protocol.json'):
            return [batch['name'],'reconcile_label_identity_without_rerolling']
        valid+=sum(r['pay_intent_filled'] for r in d['pairs'])
        nmap+=len({r['seed'] for r in d['pairs']})
    return ['aggregate','verify_identity_and_across_map_fit_support']

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--status',action='store_true');a=ap.parse_args()
    path=HERE/'next_intake.json'
    if not a.status and not path.exists():
        retain(path,build())
    if path.exists() and load(path)['source_data_sha256']!=sha(HERE/'training_labels.json'):
        raise ValueError('continuation is bound to different upstream labels')
    d=load(path)
    phase=next_phase(d)
    registered=[q.parent.name for q in (HERE/'batches').glob('intake_*/protocol.json')]
    done=[q.parent.name for q in (HERE/'batches').glob('intake_*/aggregate.json')]
    completed=[q for q in done if q in registered]
    print(json.dumps({'state_at_registration':d['state'],'registered_batches':sorted(registered),
                      'completed_batches':sorted(completed),'next':phase,'remaining_at_registration':d['remaining_minimum']}))
