"""Run the frozen held-league comparison and independently settle every command tape."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import gzip
from itertools import product
import json
import os
from pathlib import Path
from statistics import mean
import sys

import collect

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / 'evidence'
PARENT = collect.PARENT
CHILD = HERE / 'candidate.py'
ROOT = collect.ROOT
SAP = next(folder for folder in HERE.parents if (folder / 'official_game_replay.py').is_file())
sys.path.insert(0, str(SAP))
import official_game_replay
import numpy as np


def require(test, msg):
    if not test:
        raise ValueError(msg)


def read(path):
    return json.loads(path.read_bytes())


def install_record(path, obj):
    if path.exists():
        require(collect.canonical_bytes(read(path)) == collect.canonical_bytes(obj),
                f'Existing receipt differs at {path}')
    else:
        collect.write_json(path, obj)


def freeze(fit_sha):
    fit_path = HERE / 'fit.json'
    require(collect.sha_file(fit_path) == fit_sha, 'Requested fit release digest differs')
    fit = read(fit_path)
    maps = read(HERE / 'clearance.json')
    policy_plan = read(HERE / 'protocol.json')
    expected_parent = policy_plan['initial_actor']
    require(expected_parent == fit['parent_sha256'] == collect.sha_file(PARENT), 'Incorrect incumbent')
    require(fit['package_sha256'] != expected_parent and fit['package_sha256'] == collect.sha_file(CHILD),
            'Unexpected trained candidate')
    require(fit['export']['quantized_changed'] > 0 and fit['export']['package_bytes'] == CHILD.stat().st_size <= 10_000_000,
            'Wrong frozen deployable package')
    require(fit['protocol_sha256'] == collect.sha_file(HERE/'protocol.json')
            and fit['collection_sha256'] == collect.sha_file(HERE/'collection.json')
            and fit['optimizer_sha256'] == collect.sha_file(HERE/'optimize.py')
            and fit['audit_sha256'] == collect.sha_file(HERE/'decision_audit.json.gz'), 'Broken training release links')
    require(fit['physical_gate'] is True and len(fit['changed_free_joint_by_seat']) == 2
            and min(fit['changed_free_joint_by_seat']) > 0, 'Both seats need an actual changed free joint command')
    require(0 <= fit['parity']['max_signed_logit_difference_error'] <= .003
            and 0 <= fit['parity']['max_probability_error'] <= .00075, 'Training behavior audit failed')
    require(fit['sampled_episodes'] == 36 and fit['parity']['roots'] == fit['decisions'] > 0, 'Incomplete fit')
    groups = maps['groups']
    train = set(groups['train'])
    dev = sorted(groups['development'])
    original = set(groups['original_development_reserve'])
    require(len(train)==18 and len(dev)==len(set(dev))==6 and len(original)==2
            and len(set(dev)|train|original)==26 and len(set(maps['by_seed'].values()))==26,
            'Non-independent cohorts')
    require(not maps['seed_intersections'] and not maps['physical_intersections'], 'Previously exposed map')
    expected_training = {(s, seat) for s in train for seat in (0, 1)}
    require({(r['seed'], r['seat']) for r in fit['tickets']} == expected_training
            and len(fit['tickets']) == len(expected_training), 'Fit touched undeclared maps or duplicated a game')
    cards = {row['id']:row for row in read(HERE/'opponents.json')['opponents']}
    opponents = {k:{'path':v['path'], 'sha256':v['sha256']} for k,v in cards.items()}
    require(set(opponents)==set(collect.OPP_ORDER), 'Wrong opponent pool')
    for k,row in opponents.items():
        require(collect.sha_file(ROOT/row['path'])==row['sha256'], f'Wrong opponent {k}')
    plan = []
    for ix, seed in enumerate(dev):
        for seat in (0,1):
            plan.append(dict(seed=seed, seat=seat, terrain=maps['by_seed'][str(seed)],
                             opponent=collect.OPP_ORDER[ix%3],
                             order=['parent','candidate'] if (ix+seat)%2==0 else ['candidate','parent']))
    return dict(protocol_sha256=fit['protocol_sha256'], fit_sha256=fit_sha,
                source_sha256={n:collect.sha_file(HERE/n) for n in (
                    'clearance.json','opponents.json','development.py','collect.py')},
                official_replayer_sha256=collect.sha_file(SAP/'official_game_replay.py'),
                package_sha256={'parent':expected_parent, 'candidate':fit['package_sha256']}, opponents=opponents,
                gate='Positive pooled paired margin, candidate wins >= parent wins, positive mean margin delta on >=4/6 maps and >=2/3 opponent groups.',
                plan=plan)


def run_pair(arg):
    reg, ticket = arg
    seed, seat = ticket['seed'], ticket['seat']
    opponent = reg['opponents'][ticket['opponent']]
    rows = {}
    for who in ticket['order']:
        path = EVIDENCE / f'{seed}_s{seat}_{who}.json.gz'
        if path.exists():
            raw = json.loads(gzip.decompress(path.read_bytes()))
        else:
            raw = collect.play(seed, seat, ticket['terrain'], ROOT/opponent['path'],
                               PARENT if who=='parent' else CHILD, 'greedy')
            collect.write_gzip(path, raw)
        require((raw['seed'],raw['candidate_seat'],raw['arm'],raw['events'],raw['policy_sha256'],
                 raw['opponent_sha256'],raw['initial_terrain_u8_sha256']) ==
                (seed,seat,'greedy',[],reg['package_sha256'][who],opponent['sha256'],ticket['terrain']),
                f'Transcript identity mismatch {path}')
        row = collect.summarize(path, raw, ticket['opponent'])
        row.update(role=who, raw_file=path.name, raw_sha256=row['sha256'])
        checked = official_game_replay.check(EVIDENCE, row)
        require(checked['map_sha256']==ticket['terrain'], 'Officially replayed opening changed')
        row.update(official_replay=checked, local_per_player=raw['local_per_player_diagnostic'],
                   wall_s=raw['wall_s'], cash_by_seat=raw['cash_by_seat'])
        rows[who] = row
    # Prior to the first policy divergence the deterministic opponent must
    # give the same commands. Check only the shared prefix; later changes
    # can reflect genuine best responses along the two paths.
    tapes = {who:json.loads(gzip.decompress((EVIDENCE / r['raw_file']).read_bytes()))['actions'] for who,r in rows.items()}
    first = None
    for t,(p,c) in enumerate(zip(tapes['parent'], tapes['candidate'])):
        require(p[1-seat] == c[1-seat], f'Opponent diverged on a common history {seed}/{seat}/{t}')
        if p[seat] != c[seat]:
            first = t
            break
    if first is None:
        require(tapes['parent']==tapes['candidate'], 'Unexpected terminal divergence on identical actions')
    return dict(seed=seed,seat=seat,opponent=ticket['opponent'], first_changed_free_command_step=first, arms=rows)


def describe(rows):
    return dict(games=len(rows),wins=sum(r['result']=='win' for r in rows),
                draws=sum(r['result']=='draw' for r in rows),losses=sum(r['result']=='loss' for r in rows),
                mean_official_margin=mean(r['margin'] for r in rows))


def aggregate(reg, pairs):
    pairs=sorted(pairs,key=lambda r:(r['seed'],r['seat']))
    keys=[(r['seed'],r['seat'],r['opponent']) for r in pairs]
    require(keys==[(r['seed'],r['seat'],r['opponent']) for r in reg['plan']], 'Incomplete comparison')
    all_rows=[arm for pair in pairs for arm in pair['arms'].values()]
    def summary(subset):
        scores = {who:describe([r for r in subset if r['role']==who]) for who in ('parent','candidate')}
        scores['mean_margin_delta'] = scores['candidate']['mean_official_margin']-scores['parent']['mean_official_margin']
        return scores
    overall=summary(all_rows)
    per_map={key:summary([r for r in all_rows if r['seed']==key]) for key in sorted({t[0] for t in keys})}
    per_opp={key:summary([r for r in all_rows if r['opponent']==key]) for key in collect.OPP_ORDER}
    per_seat={s:summary([r for r in all_rows if r['seat']==s]) for s in (0,1)}
    deltas=np.array([x['mean_margin_delta'] for x in per_map.values()])
    bootstrap=deltas[np.asarray(list(product(range(6),repeat=6)))].mean(axis=1)
    pos_maps=sum(x>0 for x in deltas.tolist())
    pos_opp=sum(x['mean_margin_delta']>0 for x in per_opp.values())
    passed=overall['mean_margin_delta']>0 and overall['candidate']['wins']>=overall['parent']['wins'] and pos_maps>=4 and pos_opp>=2
    return dict(status='complete', pass_held_league=passed, overall=overall,
                independent_map_count=6, positive_map_count=pos_maps, positive_opponent_count=pos_opp,
                margin_delta_percentile95_map_bootstrap=np.percentile(bootstrap,[2.5,97.5]).tolist(),
                uncertainty='Descriptive paired-map resampling with enumeration of 6^6 bootstrap index vectors. These three self-owned opponents are fixed, not independently sampled opponents.',
                maps=per_map, opponents=per_opp, seats=per_seat, pairs=pairs,
                reservation_sha256=collect.sha_file(EVIDENCE/'registration.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-sha',required=True,help='Digest of explicitly released frozen fit.json')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    require(1<=args.workers<=4, 'Use 1–4 workers')
    reg=freeze(args.fit_sha)
    EVIDENCE.mkdir(exist_ok=True)
    # Directory descriptor supports interprocess exclusion without tracked temporary files.
    lock=os.open(EVIDENCE,os.O_RDONLY)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        install_record(EVIDENCE/'registration.json',reg)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_pair,(reg,t)) for t in reg['plan']]
            rows=[f.result() for f in futures]
        report=aggregate(reg,rows)
        install_record(EVIDENCE/'results.json',report)
        print(json.dumps({k:report[k] for k in ('pass_held_league','overall','positive_map_count','positive_opponent_count')},indent=2))
    finally:
        os.close(lock)


if __name__=='__main__':
    main()
