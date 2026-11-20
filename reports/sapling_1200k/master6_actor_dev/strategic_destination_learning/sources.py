"""Freeze map roles, complete historical expert tapes and time strata without inspecting scores."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
CORPUS=HERE.parent/'cargo_tree_action_learning_dev/native_actor_corpus'
EXP=CORPUS/'expansion'
ROOT=next(p for p in HERE.parents if (p/'.git').exists())
SIGNED={'inventory.jsonl.gz':'3220787149a5d54b8638d049897e598b0204734c97a44f927b6046fa73de6ae2',
        'terrains.jsonl.gz':'313a87ad222c1ce844a18c5b9993cc2258e1240a94933b750b331505b15b5d88',
        'exclusions.json':'dcff568a9854c7f918474bb34031f4788341e2c4d1efd8b75c699c1ed7c7b458'}


def sha(raw):return hashlib.sha256(raw).hexdigest()


def need(flag,why):
    if not flag:raise ValueError(why)


def rank(*items):return sha('|'.join(map(str,items)).encode())


def ticks(physical,seat,n,k):
    need(n>=k,'Insufficient complete actions for the preregistered strata')
    times=[]
    for i in range(k):
        a,b=i*n//k,(i+1)*n//k
        times.append(a+int(rank('destination-tick-v1',physical,seat,i)[:16],16)%(b-a))
    return times


def build():
    for name,value in SIGNED.items():need(sha((EXP/name).read_bytes())==value,'Source registry changed: '+name)
    def records(name):return [json.loads(row) for row in gzip.open(EXP/name)]
    maps=records('terrains.jsonl.gz');games=records('inventory.jsonl.gz')
    valid=[g for g in games if g['verified'] and g['full_official_terminal'] and g['native_action_supervision'] and not g['duplicate_trajectory_of']]
    need(len(valid)==312 and len(maps)==76,'Source cardinality differs')
    reg=json.loads((EXP/'exclusions.json').read_bytes())
    by_map=collections.defaultdict(list)
    for g in valid:by_map[g['physical_sha256']].append(g)
    older=[r for r in maps if r['pool'] in ('current_13','strict_new','retirement_outside_29') and not r['hard_protection'] and
           set(r['strict_history_exclusions'])<={'historical_development'} and max(r['seeds'])<621758000]
    # The parent weights were already trained; the eight lineage-diagnostic maps are distinct from the new fitting maps.
    held=[r for r in maps if len(r['seeds'])==1 and 621758320<=r['seeds'][0]<=621758327 and r['pool']=='retirement_outside_29' and not r['hard_protection']]
    key=lambda r:rank('destination-v1',r['physical_sha256'])
    older.sort(key=key);held.sort(key=key)
    need(len(older)==30 and len(held)==8,'Different eligible map sets')
    allocations=[('train',r) for r in older[:24]]+[('val',r) for r in held[:4]]+[('dev',r) for r in held[4:]]
    cards=[];roles=[]
    for split,r in allocations:
        physical=r['physical_sha256'];[seed]=r['seeds']
        need(reg['by_physical_sha256'][physical]==[seed] and seed not in reg['active_full_game_seed_ids'] and
             not (reg['formal_seed_half_open'][0]<=seed<reg['formal_seed_half_open'][1]) and
             not ({'published_ladder','historical_comparison_fixed'}&set(reg['by_seed'][str(seed)]['roles'])),'Protected physical terrain')
        roles.append(dict(split=split,seed=seed,terrain=physical,formerly=r['pool'],prior_roles=r['prior_roles'],
                          new_training_retirement=split=='train' and r['pool']=='retirement_outside_29',
                          prior_evidence=r['historical_role_evidence']))
        for seat in range(2):
            choices=[g for g in by_map[physical] if g['expert_seat']==seat]
            selected=min(choices,key=lambda g:rank('destination-tape-v1',physical,seat,g['path']))
            k=32 if split=='train' else 24
            cards.append(dict(seed=seed,terrain=physical,split=split,expert_seat=seat,candidate_seat=1-seat,
                              path=selected['path'],sha256=selected['raw_sha256'],transitions=selected['transitions'],
                              actions_sha256=selected['actions_sha256'],official_cash_path_sha256=selected['official_cash_path_sha256'],
                              expert_tape_sha256=selected['expert_command_tape_sha256'],opponent=selected['opponent_pack_sha256'],
                              registered_times=ticks(physical,seat,selected['transitions'],k)))
    return dict(schema='sapling-strategic-destination-v1-sources',sources=SIGNED,baseline_commit=reg['revision'],
                input_scope='312 unique complete native original games / 76 terrain hashes. Map selection uses previous roles, physical ID and seed. Between available scored tapes only path, physical ID and physical seat are used, without winner, score or player actions.',
                use='A versioned retirement is permanent for the listed historical training terrain in this model lineage. Validation and development are archived, separately held for this version. No new games or future formal maps were opened.',
                forbidden='No current native map or seed >=621758000 enters training; omit protected, older behavior holdout, reserve and prior quarantine categories, historical fixed 1000 and future formal interval. Only previously registered outside-panel seeds 320..327 used for diagnostic evaluation.',
                targets='Observe actual acting seat from t0. Previous own/foe movements from public history only. Target current physical expert order on verified legal executed prefix. For each current worker before the first invalid prefix slot, inspect same UID at t+12; survive12=0 if missing then, 1 if extant. dst12=NONE if net displacement zero, else sign of larger absolute toroidal component, horizontal if tied. Codes NONE UP RIGHT DOWN LEFT = 0..4. Full window required; all labels -1 if terminal precedes t+12. Bases, padded and blocked suffix slots are -1. Root at t+12==terminal retains final physical entities.',
                maps=roles,cards=cards)


if __name__=='__main__':
    obj=build();path=HERE/'sources.json'
    path.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(sha(path.read_bytes()),collections.Counter(c['split'] for c in obj['cards']),flush=True)
