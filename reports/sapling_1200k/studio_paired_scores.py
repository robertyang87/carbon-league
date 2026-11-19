"""Compare bigeval_live arms on the same machine and exactly the same map/seat set.

Run from the repository root after the two source machines' export manifests are available:
    python reports/sapling_1200k/studio_paired_scores.py --machine Studio \
        --baseline sqrt/origin.json --candidate v2/origin.json > comparison.json

Each arm supplies its own origin receipt, with file paths relative to that receipt:
    {"machine":"Studio", "provenance":"source host/export record or archival link",
     "examiner_source_md5":"57f9a594cce4f2ec9e3573d8d9d26ca1",
     "shards":[{"path":"bigeval_live/shard.jsonl", "sha256":"<64 hex digits>",
                "summary_path":"bigeval.jsonl", "summary_sha256":"<64 hex digits>"}]}
The summary fields are optional when only live rows are available. The output reports
how many complete shard summaries were checked. The machine designation must come from
source provenance: bigeval_live rows alone cannot distinguish Studio and spot-72.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SOURCE = '57f9a594cce4f2ec9e3573d8d9d26ca1'
WRAPPER = 'b4cde262695d38270e16bde33875a443'
HARNESS = '354bcd1bc2dc8bc188c813f53d28c893'
SQRT = 'f92b1823f3de47555f6158eecd0cfb23'
V2 = '8299bbaf060c95febd6ac9382a3b99bf'
FIELDS = {'margin': 'margin', 'own_score': 'sa', 'opponent_score': 'sb',
          'own_deposits': 'dep', 'opponent_deposits': 'odep',
          'own_tree_income': 'tinc', 'opponent_tree_income': 'otinc',
          'joint_transitions': 'actual_transitions'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(v):
    return type(v) in (int, float) and math.isfinite(v)


def read_arm(manifest_path, machine, candidate_hash, expected):
    src = manifest_path.read_bytes()
    receipt = json.loads(src)
    require(receipt['machine'] == machine and receipt['examiner_source_md5'] == SOURCE,
            f'{manifest_path}: different machine or examiner source')
    require(bool(receipt['provenance']) and bool(receipt['shards']), 'missing origin or shard list')
    result = {}
    hashes = []
    summaries_checked = 0
    for shard in receipt['shards']:
        raw = (manifest_path.parent / shard['path']).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        require(digest == shard['sha256'] and digest not in hashes, 'incorrect or repeated shard hash')
        hashes.append(digest)
        shard_full = []
        for line in raw.splitlines():
            row = json.loads(line)
            seed, side = row['seed'], row['side']
            require(type(seed) is int and type(side) is int and side == seed % 2, 'invalid seed/seat')
            require(seed in expected and seed not in result, f'{seed}: unexpected or repeated map')
            require((row['ah'], row['bh'], row['hh']) == (candidate_hash, WRAPPER, HARNESS),
                    f'{seed}: candidate, examiner wrapper, or harness mismatch')
            require(row['err'] is None and row['serr'] == row['berr'] == 0,
                    f'{seed}: engine or agent error')
            a, b, m = row['sa'], row['sb'], row['margin']
            require(all(number(v) for v in (a,b,m)) and abs(a-b-m) <= 1e-6, f'{seed}: score mismatch')
            require(type(row['win']) is bool and row['win'] == (m > 0), f'{seed}: win mismatch')
            win_value = float(m > 0) + 0.5 * (m == 0)
            utility = 0.5 * win_value + 0.5 / (1 + math.exp(-m/1000))
            require(number(row['u']) and abs(row['u']-utility) <= 1e-12, f'{seed}: invalid U')
            require(not any(row.get(k) not in (None, machine) for k in ('machine', 'hostname')),
                    f'{seed}: live row contradicts machine attestation')
            times = [row[k] for k in ('end_step', 'steps_played', 'actual_transitions') if row.get(k) is not None]
            if times:
                require(all(type(n) is int and n == times[0] for n in times), f'{seed}: inconsistent steps')
                row['actual_transitions'] = times[0]
            for name in FIELDS.values():
                if row.get(name) is not None:
                    require(number(row[name]), f'{seed}: invalid {name}')
            if row.get('actual_transitions') is not None:
                require(type(row['actual_transitions']) is int and 1 <= row['actual_transitions'] <= 299,
                        f'{seed}: invalid transition count')
            result[seed] = row
            shard_full.append([seed,side,m,a,b,row['u']])
        if 'summary_path' in shard:
            summary_bytes = (manifest_path.parent / shard['summary_path']).read_bytes()
            require(hashlib.sha256(summary_bytes).hexdigest() == shard['summary_sha256'], 'summary hash differs')
            summary = json.loads(summary_bytes)
            health = ('err','search_error_games','a_errors_total','root_errors_total','b_search_error_games',
                      'b_errors_total','b_root_errors_total')
            require(summary['FAIL'] is False and all(summary[k] == 0 for k in health), 'failed exam summary')
            require((summary['a_md5'],summary['b_md5'],summary['harness_md5']) == (candidate_hash, WRAPPER, HARNESS),
                    'summary identity mismatch')
            expected_shard = set(range(summary['seed0'], summary['seed0'] + summary['n']))
            require(expected_shard == {r[0] for r in shard_full} and summary['n'] == len(shard_full),
                    'summary map range/count mismatch')
            require(sorted(summary['games_full']) == sorted(shard_full) and
                    sorted(summary['games']) == sorted(r[:3] for r in shard_full), 'summary per-game records differ')
            require(summary['wins'] == sum(r[2] > 0 for r in shard_full), 'summary win count differs')
            summaries_checked += 1
    require(result.keys() == expected, f'{manifest_path}: missing maps {sorted(expected-result.keys())[:8]}')
    provenance = {'machine': machine, 'attestation': receipt['provenance'],
                  'manifest_sha256': hashlib.sha256(src).hexdigest(), 'shard_sha256': hashes,
                  'candidate_md5': candidate_hash, 'checked_complete_shard_summaries': summaries_checked}
    return result, provenance


def summarize(base, new, seeds, rng):
    stats = {}
    for label, field in FIELDS.items():
        use = [s for s in seeds if base[s].get(field) is not None and new[s].get(field) is not None]
        if not use:
            stats[label] = {'map_pairs': 0, 'missing_pairs': len(seeds)}
            continue
        a = np.asarray([base[s][field] for s in use], dtype=np.float64)
        b = np.asarray([new[s][field] for s in use], dtype=np.float64)
        d = b-a
        strata = [d[np.array(use)%2 == side] for side in (0, 1)]
        strata = [cell for cell in strata if len(cell)]
        ci = None
        if all(len(cell) > 1 for cell in strata):
            resampled = sum(rng.choice(cell, (5000, len(cell)), replace=True).sum(1) for cell in strata)
            ci = np.quantile(resampled/len(use), [.025, .975]).tolist()
        stats[label] = {'map_pairs':len(use), 'missing_pairs':len(seeds)-len(use), 'base':float(a.mean()),
                        'new':float(b.mean()), 'delta':float(d.mean()), 'paired_bootstrap_95':ci}
    return {'maps':len(seeds), 'base_wins':sum(base[s]['margin']>0 for s in seeds),
            'new_wins':sum(new[s]['margin']>0 for s in seeds), 'fields':stats}


def compare(baseline, candidate, start, count, machine, base_md5=SQRT, new_md5=V2):
    require(count>0 and base_md5 != new_md5, 'expected positive count and distinct candidate identities')
    seeds = set(range(start, start+count))
    base, a = read_arm(baseline, machine, base_md5, seeds)
    new, b = read_arm(candidate, machine, new_md5, seeds)
    ordered = sorted(seeds)
    rng = np.random.default_rng(714817)
    output = {'method':'One game per map/arm, same map and seat. 5,000 paired map bootstrap samples within seat. '
                     'Unseeded examiner trajectories may differ. bigeval_live has no hostname; origin relies on the '
                     'source attestation bound to each shard SHA. Optional transitions are read only from an explicit count.',
              'baseline':a, 'candidate':b, 'first_seed':start, 'count':count,
              'fixed_examiner_source_md5':SOURCE, 'fixed_wrapper_md5':WRAPPER, 'fixed_harness_md5':HARNESS,
              'overall':summarize(base,new,ordered,rng), 'by_seat':{}, 'by_map':[]}
    for side in (0,1):
        ss = [s for s in ordered if s%2==side]
        if ss:
            output['by_seat'][str(side)] = summarize(base,new,ss,rng)
    for seed in ordered:
        values = {label:(new[seed][field]-base[seed][field]) if new[seed].get(field) is not None and
                  base[seed].get(field) is not None else None for label,field in FIELDS.items()}
        output['by_map'].append({'seed':seed, 'side':seed%2, 'new_minus_base':values})
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True, help='baseline origin manifest')
    p.add_argument('--candidate', type=Path, required=True, help='candidate origin manifest')
    p.add_argument('--machine', required=True, help='exact machine identifier on both manifests')
    p.add_argument('--start', type=int, default=20291101)
    p.add_argument('--count', type=int, default=200)
    p.add_argument('--baseline-md5', default=SQRT)
    p.add_argument('--candidate-md5', default=V2)
    args = p.parse_args()
    print(json.dumps(compare(args.baseline, args.candidate, args.start, args.count, args.machine,
                             args.baseline_md5, args.candidate_md5), indent=2, allow_nan=False))
