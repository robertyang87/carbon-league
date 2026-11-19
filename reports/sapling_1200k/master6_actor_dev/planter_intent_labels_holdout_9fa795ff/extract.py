"""Apply the frozen worker-label definitions to the five new paid-game records and certify them."""
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'planter_intent_labels'
TOP = HERE.parents[1] / 'all_top5'
BATCH = TOP / 'appendix_9fa795ff'
ROOT = HERE.parents[3]
sys.path.insert(0, str(ORIGINAL))
import source
import verify

PINNED = {
    ORIGINAL / 'source.py': '5514c001651b01c333c86d710464836c2d731812764484edcb5a384701519b18',
    ORIGINAL / 'schema.json': 'a2e30c15609c2d4d2b478a879d6cf3295abc0ab8f53f96593fad22ca73fa4523',
    ORIGINAL / 'verify.py': '6471e056c7e8f54e79981033e2aa4d1cddeecd6c1c1f3fe0adf07588ef27410c',
    ORIGINAL / 'maps.json.gz': 'e8cedbdba5c0ad1aa27f0c1206a8ed885800dde5fb5e1aaeb84bb360f11e15db',
    BATCH / 'manifest.json': '37a2a84add511752adf5caa17a5f974d619630e37dc96991c67d22c3b4489625',
    BATCH / 'cases.json.gz': '9610f1d737a1b43dc1d790916eb6af1e29dccd443702d906357f6c3d12026f6e',
    BATCH / 'validation_maps.jsonl.gz': '8a9bc8be7fad62298ad9af45490d12900f02a9d770a622280b4f77de99f4b1c4',
    BATCH / 'validation.json': '2f3d8e0c8880d9a7702cbb4c542948b56b19a6ffe11e0d4a02de2dcfdab960f2',
}
EXAMPLES = {'paid_by_intra_turn_cash','paid_under_pre_cash','stop_no_pay_survived',
            'stop_no_pay_died','moving_survived','moving_died'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(ok, failure):
    if not ok:
        raise ValueError(failure)


def main():
    check(all(sha(p) == v for p, v in PINNED.items()), 'source/schema/cash ledger changed')
    manifest = source.read(BATCH / 'manifest.json')
    ground = source.read(BATCH / 'validation.json')
    validation = {x['id']: x for x in map(json.loads, gzip.open(BATCH / 'validation_maps.jsonl.gz','rt'))}
    cash = {g['id']: g for g in source.read(BATCH / 'cases.json.gz')['rows']}
    old = source.read(ORIGINAL / 'maps.json.gz')['maps']
    cards = manifest['games']
    check(len(cards)==len(cash)==len(validation)==5 and len({g['id'] for g in cards})==5 and
          manifest['revision']=='9fa795ffa2c65502e75cd708afcb4b2e96f6778f' and ground['score_cards_agree']==5
          and ground['fastsim_transitions']==ground['original_board_transitions']==1495, 'wrong new cohort')
    maps, data, seats, evidence, examples = [], [], [], [], {}
    counts = Counter()
    for index, g in enumerate(cards):
        gid = g['id']
        check(g['frame_verified'] and cash[gid]['source_sha256']==g['sha256'] and
              cash[gid]['planting']['rewards']==g['score_by_seat'], 'wrong replay or cash ledger')
        loc = g['archive_location']
        m = dict(id=gid, map_index=index, seed=g['seed'], names=g['seat_names'], seat_strategy_ids=g['seat_strategy_ids'],
                 terrain=validation[gid]['initial_terrain_u8_sha256'], sha256=g['sha256'], turns=g['transitions'],
                 archive=loc['archive'],member=loc['member'],git_path=None,git_blob=None,split='external_9fa795ff')
        maps.append(m)
        check(all(all(m[k] != x[k] for k in ('id','terrain','sha256','seed')) for x in old+maps[:-1]),
              'overlap with a previous map')
        columns = [[] for k in source.NAMES]
        c = Counter()
        source.game(m,source.CACHE,columns,c)
        a = {k: np.asarray(v, dtype=source.TYPES[k]) for k,v in zip(source.NAMES,columns)}
        verify.numeric(a)
        sample, implicit, explicit = verify.raw_check(m, g, a, source.CACHE.parent, EXAMPLES)
        check(implicit+explicit==c['stop'], 'raw stop denominator mismatch')
        for k,v in sample.items():
            examples.setdefault(k,v)
        data.append(a)
        ledger = {(t,s,uid):(cell,int(kind=='seize'),price)
                  for t,s,uid,_,cell,kind,price,*_ in cash[gid]['planting']['purchases']}
        charged = {}
        for j in np.flatnonzero(a['y_paid']):
            seat,worker,t = (int(a[k][j]) for k in ('seat','worker_number','t'))
            k = (t,seat,f'player-{seat}-worker-{worker}')
            charged[k] = (int(a['cell'][j]), int(a['kind'][j]), float(a['y_actual_price'][j]))
        check(len(ledger)==len(cash[gid]['planting']['purchases']) and ledger.keys()==charged.keys() and
              len(charged)==int(a['y_paid'].sum()) and all(ledger[k][:2]==v[:2] and abs(ledger[k][2]-v[2])<=1e-5 for k,v in charged.items()),
              'settled per-worker payment differs from independent ledger')
        evidence.append(dict(id=gid,cash_events=len(charged),implicit_stops=implicit,explicit_stops=explicit))
        for seat in (0,1):
            own = {k:v[a['seat']==seat] for k,v in a.items()}
            row = verify.count_fields(own)
            seats.append(dict(id=gid,map_index=index,seat=seat,name=m['names'][seat],**row))
            counts.update(row)
    check(set(examples)==EXAMPLES, 'positive/negative classes incomplete')
    shard = HERE / 'labels.npz'
    arrays = {k:np.concatenate([a[k] for a in data]) for k in source.NAMES}
    if shard.exists():
        with np.load(shard,allow_pickle=False) as f:
            check(set(f.files)==set(arrays) and all(f[k].dtype==a.dtype and np.array_equal(f[k],a) for k,a in arrays.items()),
                  'frozen array differs from regenerated holdout')
    else:
        stage=shard.with_name('labels.partial.npz'); np.savez_compressed(stage, **arrays); os.replace(stage, shard)
    source.json_save(HERE/'maps.json', dict(schema_version=1,revision=manifest['revision'],maps=maps))
    source.json_save(HERE/'verification.json',dict(schema_version=1, scope='Frozen temporal external holdout, both physical seats, no fit or threshold selection.',
        schema_path=str((ORIGINAL/'schema.json').relative_to(ROOT)), seats=seats,counts=dict(counts),includes_every_new_map=True,
        original_steps_independently_reconciled=ground['original_board_transitions'],source_replay_checks=evidence,
        original_replay_examples=examples,files_sha256={str(p.relative_to(ROOT)):sha(p) for p in [*PINNED,Path(__file__),shard,HERE/'maps.json']}))
    print(dict(counts), sha(HERE/'verification.json'))


if __name__ == '__main__':
    main()
