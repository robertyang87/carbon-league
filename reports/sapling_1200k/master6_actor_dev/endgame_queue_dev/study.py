"""Replay sealed original actions, then compare public endgame choices at the two registered turns."""
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = BASE.parents[1]
ENGINE = ROOT / 'code/env_src/carbon_challenge-master'
sys.path[:0] = [str(BASE), str(ROOT / 'code'), str(ENGINE)]
from tree_win_gate import validate_examiner_configuration, validate_game, verify_raw
from turning_point_ledger import CashEvents, FastCarbon
from zerosum_env.envs.carbon import idgen
from zerosum_env.envs.carbon.carbon import compute_next_board
from zerosum_env.utils import structify

PROTOCOL = '575159ddc7aceb0113ec69518cbf172cde7f2dba3a9a8abac5b2e81c284c9f44'
REPLAY = HERE / 'replay_roots.json.gz'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(path.read_bytes())


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def write(path, obj):
    raw = json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=None if path.suffix == '.gz' else 2).encode()
    data = gzip.compress(raw, compresslevel=5, mtime=0) if path.suffix == '.gz' else raw + b'\n'
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
        out.write(data)
    os.replace(out.name, path)


@lru_cache(maxsize=None)
def producer(name, expected):
    path = ROOT / name
    if path.is_file() and sha(path) == expected:
        return 'worktree'
    # The checked-in map exclusion registry has expanded since collection.
    history = subprocess.check_output(['git', 'log', '--format=%H', '--', name], cwd=ROOT, text=True).splitlines()
    for rev in history:
        payload = subprocess.run(['git', 'show', f'{rev}:{name}'], cwd=ROOT, capture_output=True)
        if payload.returncode == 0 and digest(payload.stdout) == expected:
            return rev
    raise ValueError(f'cannot reproduce arena source {name} {expected}')


def protocol():
    require(sha(HERE / 'protocol.json') == PROTOCOL, 'protocol changed')
    doc = read(HERE / 'protocol.json')
    require([s['batch'] for s in doc['source_batches']] == list(range(1, 8)) and doc['roots'] == [297, 298], 'allocation')
    return doc


def batch_files(spec, registered):
    receipt = ROOT / spec['receipt']
    area, src = receipt.parent, receipt.parent / 'source'
    require(sha(receipt) == spec['receipt_sha256'], 'source receipt changed')
    signed = read(receipt)
    official, native = read(src / 'official_replay.json'), read(src / 'results.json')
    require(sha(src / 'official_replay.json') == spec['official_sha256'] == signed['official_sha256'], 'official seal')
    for filename in ('protocol', 'clearance'):
        require(sha(area / f'{filename}.json') == signed[f'{filename}_sha256'], filename)
    for name, value in official['engine_source_sha256'].items():
        require(sha(ENGINE / name) == value, 'replayer changed: ' + name)
    head = native['header']
    pinned = head['source_sha256']
    sources = {k: {'sha256': v, 'verified_at': producer(k, v)} for k, v in pinned.items()}
    require(pinned['data/master3s2_source.py'] == registered['source_examiner'], 'examiner')
    require(native['status'] == 'complete' and head['pack_sha256'] == official['runs']['source']['pack_sha256'] ==
            registered['source_parent'] and head['weight_sha256'] == read(area / 'protocol.json')['parent_weight_manifest_sha256'], 'package')
    groups = []
    for collection, seat in [(spec['games'], 'seat'), (signed['games'], 'seat'), (native['games'], 'candidate_seat'),
                             (official['runs']['source']['games'], 'seat')]:
        group = {(r['seed'], r[seat]): r for r in collection}
        require(len(group) == len(collection) == 4, 'duplicate or missing game')
        groups.append(group)
    expected = {(seed, seat) for seed in head['seeds'] for seat in (0, 1)}
    require(all(set(g) == expected for g in groups), 'map/seat allocation')
    batch = {'batch': spec['batch'], 'receipt_sha256': sha(receipt), 'official_sha256': sha(src / 'official_replay.json'),
             'native_sha256': sha(src / 'results.json'), 'arena_sources': sources, 'official_engine': official['engine_source_sha256']}
    games = []
    for key in sorted(expected):
        planned, row, actual, off = [g[key] for g in groups]
        path = ROOT / row['raw_path']
        require(path == src / actual['raw_file'] and sha(path) == planned['raw_sha256'] == row['raw_sha256'] ==
                actual['raw_sha256'] == off['transcript_sha256'], 'tape identity')
        require(off['statuses'] == ['DONE','DONE'] and off['map_sha256'] == row['map_sha256'] == planned['terrain']
                and actual['actual_transitions'] == off['steps'] == row['steps'] and actual['official_score_by_seat'] ==
                off['official_reward_by_seat'] == row['scores'], 'terminal signature')
        validate_game(actual)
        verify_raw(src, actual, head['pack_sha256'], head['weight_sha256'], expected_terrain=planned['terrain'])
        games.append(row)
    return batch, games


def twin_step(sim, joint, *, events=True):
    before = sim.full_observation()
    # idgen uses process-global counters. Every fork receives the precise fast simulator's state.
    idgen.reset()
    for seat in (0, 1):
        idgen._id_dict[f'player-{seat}-worker-'] = sim.worker_counters[seat]
        idgen._id_dict[f'player-{seat}-tree-'] = sim.tree_counters[seat]
    states = structify([{'observation': dict(before, full_carbon=before['carbon']) if p == 0 else {'player': 1},
                         'reward': before['players'][p][0], 'status': 'ACTIVE', 'action': joint[p]} for p in (0, 1)])
    off = compute_next_board(states, structify({'configuration': sim.config}))
    sim.cash = CashEvents(sim.cash) if events else list(sim.cash)
    sim.step(*joint)
    after = sim.full_observation()
    require(after['players'] == off[0].observation.players and after['trees'] == off[0].observation.trees and
            after['carbon'] == off[0].observation.full_carbon and after['step'] == off[0].observation.step, 'official vs fast board')
    require(all(sim.obs(p)['carbon'] == off[p].observation.carbon for p in (0, 1)), 'visibility mismatch')
    status = [a.status for a in off]
    if sim.step_count >= sim.config['episodeSteps'] - 1:
        # Environment.step applies this cap after computing rewards.
        status = ['DONE' if s in ('ACTIVE', 'INACTIVE') else s for s in status]
    account = {'cash': list(sim.cash), 'score': [a.reward for a in off], 'status': status}
    if events:
        account['events'] = sim.cash.events
        flow = [Counter(), Counter()]
        for kind, p, amount, *_ in sim.cash.events:
            flow[p][kind] += amount
        require(all(abs(sum(flow[p].values()) - (sim.cash[p] - before['players'][p][0])) < 1e-6 for p in (0,1)), 'cash ledger')
        account['flow'] = [dict(c) for c in flow]
        account['deaths'] = [sorted(set(before['players'][p][2]).difference(after['players'][p][2])) for p in (0,1)]
        account['collisions'] = sim.last_collisions
        account['inventory'] = [sum(w[1] for w in pl[2].values()) for pl in after['players']]
    return account


def prepare():
    doc = protocol()
    bundles, captures = [], []
    for spec in doc['source_batches']:
        meta, games = batch_files(spec, doc)
        bundles.append(meta)
        for row in games:
            raw = read(ROOT / row['raw_path'])
            validate_examiner_configuration(raw['configuration'])
            require(not raw['audit_messages'] and not raw['errors'], 'invalid raw tape')
            cfg = raw['game_configuration']
            sim = FastCarbon(cfg).reset(seed=row['seed'])
            require(digest(bytes(int(v) for v in sim.carbon)) == row['map_sha256'] and list(sim.carbon) == raw['initial_full_carbon'], 'terrain regeneration')
            history, snapshots, orders = {}, {}, {}
            for t, joint in enumerate(raw['actions']):
                if t >= 295:
                    history[t] = [sim.obs(p) for p in (0, 1)]
                if t in doc['roots']:
                    snapshots[t] = sim.get_state()
                    orders[t] = joint
                outcome = twin_step(sim, joint, events=False)
                require(outcome['status'] == (['DONE','DONE'] if t == len(raw['actions']) - 1 else ['ACTIVE','ACTIVE']), 'unexpected termination')
            require(outcome['score'] == raw['official_score_by_seat'] == row['scores'], 'official terminal score mismatch')
            excluded = row['steps'] < sim.config['episodeSteps'] - 1
            if excluded:
                require(any(raw['extinct_by_seat']), 'unexplained short tape')
            captures.append({'batch': spec['batch'], 'split': spec['role'], 'seed': row['seed'], 'seat': row['seat'],
                             'raw_sha256': row['raw_sha256'], 'terrain': row['map_sha256'], 'steps': row['steps'],
                             'excluded': excluded, 'original': outcome, 'config': cfg, 'history': history,
                             'states': snapshots, 'orders': orders})
        print(f"verified batch {spec['batch']} ({len(captures)} games)", flush=True)
    write(REPLAY, {'protocol_sha256': PROTOCOL, 'batches': bundles, 'games': captures})


def clean(commands):
    return {uid: value for uid, value in commands.items() if value not in (None, 'NONE')}


def cash_delta(left, right):
    return [round(float(b - a), 6) for a, b in zip(left, right)]


def difference(baseline, alt, seat):
    cash = cash_delta(baseline['cash'], alt['cash'])
    score = cash_delta(baseline['score'], alt['score'])
    by_seat = []
    for p in (0, 1):
        a, b = baseline['flow'][p], alt['flow'][p]
        by_seat.append({k: round(b.get(k, 0) - a.get(k, 0), 6) for k in sorted(a.keys() | b.keys())})
    return {'cash': cash, 'score': score, 'cash_margin': round(cash[seat] - cash[1-seat], 6),
            'score_margin': round(score[seat] - score[1-seat], 6), 'flows_by_seat': by_seat,
            'worker_death_count': cash_delta(list(map(len, baseline['deaths'])), list(map(len, alt['deaths']))),
            'held_cargo': cash_delta(baseline['inventory'], alt['inventory'])}


def order_differences(joint, recorded):
    out = []
    for a, b in zip(map(clean, joint), map(clean, recorded)):
        out.append({k: [b.get(k), a.get(k)] for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)})
    return out


def evaluate_root(game, t, policy):
    seat, cfg = game['seat'], game['config']
    public_cfg = {k: v for k, v in cfg.items() if k != 'randomSeed'}
    state = game['states'][str(t)]
    source = game['orders'][str(t)]
    hist = game['history']
    public = hist[str(t)]
    past = {int(step): observations[seat] for step, observations in hist.items() if int(step) < t}
    own = source[seat]
    candidate, audit = policy.choose(public[seat], past, dict(own), public_cfg)
    own_diff = order_differences([candidate], [own])[0]
    require(set(own_diff) == set(audit['edits']) and all(k in public[seat]['players'][seat][2] for k in own_diff), 'audit/selected edit mismatch')
    sim = FastCarbon(cfg)
    sim.set_state(state)
    require(all(sim.obs(p) == public[p] for p in (0, 1)), 'restored public state drift')
    first = [[], []]
    for alt in (False, True):
        joint = list(source)
        if alt:
            joint[seat] = candidate
        if alt and not own_diff:
            first[1] = first[0]
            continue
        sim.set_state(state)
        account = twin_step(sim, joint)
        first[int(alt)] = account, sim.get_state()
    a, b = (f[0] for f in first)
    row = {k: game[k] for k in ('batch','split','seed','seat','raw_sha256')}
    row.update(t=t, changed=bool(own_diff), audit=audit, changes=own_diff, recorded_joint=source,
               candidate_own=candidate, baseline=a, alternative=b, first_delta=difference(a,b,seat), continuation=[])
    if t == 298:
        require(a['score'] == game['original']['score'] and a['status'] == ['DONE','DONE'], 'one-step factual reproduction')
        return row
    next_actions = game['orders']['298']
    sim.set_state(first[0][1])
    source_terminal = twin_step(sim, next_actions)
    require(source_terminal['score'] == game['original']['score'] and source_terminal['status'] == ['DONE','DONE'], 'two-step factual reproduction')
    row['recorded_second_joint'] = next_actions
    row['original_factual_terminal'] = source_terminal
    for variant in range(3):
        arms = []
        for i, (first_result, child) in enumerate(first):
            if i and not own_diff:
                arms.append(arms[0])
                continue
            if first_result['status'] == ['DONE','DONE']:
                arms.append({'joint': None, 'terminal': first_result, 'early_terminal': True})
                continue
            sim.set_state(child)
            joint = [policy.orders(sim.obs(p), {t: public[p]}, public_cfg, variant) for p in (0,1)]
            end = twin_step(sim, joint)
            require(end['status'] == ['DONE','DONE'], 'nonterminal continuation')
            arms.append({'joint': joint, 'terminal': end, 'early_terminal': False,
                         'moves_vs_recorded': order_differences(joint, next_actions),
                         'terminal_score_vs_original': cash_delta(game['original']['score'], end['score'])})
        stats = difference(arms[0]['terminal'], arms[1]['terminal'], seat)
        totals = []
        for p in (0,1):
            def combined(i):
                total = Counter(first[i][0]['flow'][p])
                if not arms[i]['early_terminal']:
                    total.update(arms[i]['terminal']['flow'][p])
                return total
            l, r = combined(0), combined(1)
            totals.append({k: round(r[k]-l[k], 6) for k in sorted(l.keys() | r.keys())})
            require(abs(sum(totals[p].values())-stats['cash'][p]) < 0.0001, 'two-step cash attribution')
        row['continuation'].append({'variant': variant, 'baseline': arms[0], 'alternative': arms[1], 'terminal_delta': stats,
                                    'two_step_flow_delta_by_seat': totals,
                                    'changed_responses_by_seat': order_differences(arms[1]['joint'], arms[0]['joint'])
                                    if all(x['joint'] is not None for x in arms) else None})
    return row


def summarize(rows, games):
    output = {}
    for split in ('diagnostic', 'evaluation'):
        source = [g for g in games if g['split'] == split]
        subset = [r for r in rows if r['split'] == split]
        result = {'source_games': len(source), 'source_maps': len({g['seed'] for g in source}),
                  'early_extinct_games': [{k:g[k] for k in ('seed','seat','steps')} for g in source if g['excluded']],
                  'registered_roots': 2*len(source), 'observed_eligible_roots': len(subset), 'horizons': {}}
        for t in (297, 298):
            group = [r for r in subset if r['t'] == t]
            def stats(values):
                maps = defaultdict(list)
                for r,v in zip(group, values):
                    maps[r['seed']].append(v)
                return {'sum': round(sum(values), 6), 'positive': sum(v > 0.00001 for v in values),
                        'negative': sum(v < -0.00001 for v in values), 'zero': sum(abs(v) <= 0.00001 for v in values),
                        'map_mean': round(statistics.mean(statistics.mean(v) for v in maps.values()), 6) if maps else None,
                        'per_map': {str(s): round(statistics.mean(v), 6) for s,v in maps.items()}}
            h = {'roots': len(group), 'edits': sum(r['changed'] for r in group), 'seeds': len({r['seed'] for r in group}),
                 'reasons': dict(Counter(r['audit']['reason'] for r in group)),
                 'loaded_shared_home_before': sum(any(b['inbound_loaded'] >= 2 for b in r['audit']['parent_shared_bank']) for r in group),
                 'edits_with_loaded_shared_home_after': sum(r['changed'] and any(b['inbound_loaded'] >= 2 for b in r['audit']['selected_shared_bank']) for r in group),
                 'cash_margin_first': stats([r['first_delta']['cash_margin'] for r in group]),
                 'score_margin_first': stats([r['first_delta']['score_margin'] for r in group]),
                 'choice_cpu_ms_total': round(sum(r['audit']['cpu_ms'] for r in group), 3)}
            if t == 297:
                variants = {}
                for k in range(3):
                    calls = [r['continuation'][k] for r in group]
                    score = stats([c['terminal_delta']['score_margin'] for c in calls])
                    score['baseline_proxy_score_differs_from_original'] = sum(any(abs(v)>0.00001 for v in c['baseline']['terminal_score_vs_original']) for c in calls)
                    score['baseline_proxy_different_orders_by_physical_seat'] = [sum(bool(c['baseline']['moves_vs_recorded'][p]) for c in calls) for p in (0,1)]
                    variants[k] = score
                h['conditional_terminal_scenarios'] = variants
            result['horizons'][str(t)] = h
        output[split] = result
    return output


def evaluate():
    import policy
    require(Path(policy.__file__).resolve() == HERE/'policy.py', 'wrong public selector module')
    doc = protocol()
    replay = read(REPLAY)
    require(replay['protocol_sha256'] == PROTOCOL and len(replay['games']) == 28, 'incorrect replay cache')
    rows = []
    for spec, batch in zip(doc['source_batches'], replay['batches'], strict=True):
        require(spec['receipt_sha256'] == batch['receipt_sha256'] and spec['official_sha256'] == batch['official_sha256'], 'stale batch cache')
        for ref in spec['games']:
            matches = [g for g in replay['games'] if (g['seed'],g['seat'])==(ref['seed'],ref['seat'])]
            require(len(matches)==1 and matches[0]['raw_sha256']==ref['raw_sha256'] and matches[0]['batch']==spec['batch'], 'wrong cached game')
            game = matches[0]
            if not game['excluded']:
                for t in doc['roots']:
                    rows.append(evaluate_root(game, t, policy))
    output = HERE/'root_audit.json.gz'
    write(output,rows)
    summary = summarize(rows,replay['games'])
    write(HERE/'study_results.json', {'protocol_sha256': PROTOCOL, 'replayed_turns':sum(g['steps'] for g in replay['games']),
          'complete': True, 'bindings': {p.name: sha(p) for p in (Path(__file__), HERE/'policy.py', REPLAY, output,
                     ROOT/'code/fastsim.py', BASE/'turning_point_ledger.py', BASE/'tree_win_gate.py')}, 'outcomes': summary})
    print(json.dumps({s:{t:(h['roots'],h['edits'],h['cash_margin_first']['map_mean']) for t,h in v['horizons'].items()} for s,v in summary.items()}), flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    if args not in ([], ['prepare'], ['evaluate']):
        raise SystemExit('usage: study.py [prepare|evaluate]')
    if args != ['evaluate']:
        prepare()
    if args != ['prepare']:
        evaluate()
