"""Parse Carbon replays into per-player behavior metrics — hardened v2.

Same metric definitions as parse.py (regression-verified value-for-value);
v2 adds, without changing any existing definition:
- output rows carry match id / created / first-move flag, plus trees@100/150/200/250
- per-step series export (<out>.series.json.gz): cash, live trees, workers,
  cargo, planters, collectors for both players (None on steps with no observation)
- no silent skips: every skipped match is reported with a reason
  (replay-missing / mapping-mismatch / mapping-ambiguous) in the console
  summary and in <out>.skips.json
- observation-shape accounting per game (steps with 0 or >1 observation events)
- replay lookup uses the match's own `game` number before legacy fallbacks

Usage: python3 parse2.py <matches.json> <target_name> <replay_dir> <out.json>
"""
import gzip, json, os, sys


def load_replay(path):
    if path.endswith('.gz'):
        return json.load(gzip.open(path, 'rt'))
    return json.load(open(path))


def find_replay(rdir, m):
    mid, game = m['id'], m.get('game', 1)
    cands = [f'{mid}-{game}.json.gz', f'{mid}-{game}.json',
             f'{mid}.json', f'{mid}.json.gz',
             f'{mid}-1.json.gz', f'{mid}-1.json']
    seen = set()
    for cand in cands:
        if cand in seen:
            continue
        seen.add(cand)
        p = os.path.join(rdir, cand)
        if os.path.exists(p):
            return p
    return None


def parse_game(d):
    """Return {0: metrics+series, 1: metrics+series} from a replay dict.

    Metric accumulation is byte-for-byte the parse.py logic; the series and
    the observation-shape counters are additive only."""
    steps = d['steps']
    n = len(steps)
    acc = {p: {'trees_seen': set(), 'workers_seen': set(), 'tree_steps': 0,
               'cash': {}, 'end_trees': 0, 'end_workers': 0, 'peak_cargo': 0.0,
               'end_cargo': 0.0, 'first_plant': None,
               'series': {'cash': [], 'trees': [], 'workers': [], 'cargo': [],
                          'planters': [], 'collectors': []},
               'obs0': 0, 'obs_multi': 0} for p in (0, 1)}
    checkpoints = {50, 100, 200, 269, 270, n - 1}
    for s, step in enumerate(steps):
        snap = None
        n_obs = 0
        for e in step:
            obs = e.get('observation') or {}
            if 'players' in obs:
                n_obs += 1
                if snap is None:
                    snap = obs
        for p in (0, 1):
            a = acc[p]
            if n_obs == 0:
                a['obs0'] += 1
            if n_obs > 1:
                a['obs_multi'] += 1
            if snap is None:
                for k in a['series']:
                    a['series'][k].append(None)
                continue
            pl = snap['players'][p]
            cash, workers, trees = pl[0], pl[2], pl[3]
            before = len(a['trees_seen'])
            a['trees_seen'] |= set(trees.keys())
            if a['first_plant'] is None and len(a['trees_seen']) > before:
                a['first_plant'] = s
            a['workers_seen'] |= set(workers.keys())
            a['tree_steps'] += len(trees)
            if s in checkpoints or s == n - 1:
                a['cash'][s] = cash
            cargo = sum(v[1] for v in workers.values())
            a['peak_cargo'] = max(a['peak_cargo'], cargo)
            if s == n - 1:
                a['end_trees'] = len(trees)
                a['end_workers'] = len(workers)
                a['end_cargo'] = cargo
            ser = a['series']
            ser['cash'].append(cash)
            ser['trees'].append(len(trees))
            ser['workers'].append(len(workers))
            ser['cargo'].append(cargo)
            ser['planters'].append(sum(1 for v in workers.values()
                                       if len(v) > 2 and v[2] == 'PLANTER'))
            ser['collectors'].append(sum(1 for v in workers.values()
                                         if len(v) > 2 and v[2] == 'COLLECTOR'))
    out = {}
    for p in (0, 1):
        a = acc[p]
        cash = a['cash']
        ser = a['series']
        out[p] = {
            'plants': len(a['trees_seen']),
            'tree_steps': a['tree_steps'],
            'end_trees': a['end_trees'],
            'cash50': cash.get(50), 'cash100': cash.get(100),
            'cash200': cash.get(200), 'cash270': cash.get(270),
            'cash_final': cash.get(n - 1),
            'last30': (cash.get(n - 1) - cash.get(269)) if (n - 1) in cash and 269 in cash else None,
            'recruits': len(a['workers_seen']),
            'deaths': len(a['workers_seen']) - a['end_workers'],
            'end_workers': a['end_workers'],
            'peak_cargo': a['peak_cargo'], 'end_cargo': a['end_cargo'],
            'first_plant': a['first_plant'],
            'final_reward': d['rewards'][p],
            'trees100': ser['trees'][100] if n > 100 else None,
            'trees150': ser['trees'][150] if n > 150 else None,
            'trees200': ser['trees'][200] if n > 200 else None,
            'trees250': ser['trees'][250] if n > 250 else None,
            '_series': ser,
            '_obs0': a['obs0'], '_obs_multi': a['obs_multi'],
        }
    return out


def main():
    matches_path, target, rdir, out_path = sys.argv[1:5]
    matches = json.load(open(matches_path))
    rows, series, skips = [], {}, []
    mapped_ok, mapped_bad = 0, 0
    for m in matches:
        rp = find_replay(rdir, m)
        if not rp:
            skips.append({'id': m['id'], 'reason': 'replay-missing'})
            continue
        d = load_replay(rp)
        ra, rb = d['rewards']
        sa, sb = m['scores']['a'], m['scores']['b']
        ok_fwd = abs(ra - sa) < 1.0 and abs(rb - sb) < 1.0
        ok_rev = abs(ra - sb) < 1.0 and abs(rb - sa) < 1.0
        if ok_fwd and ok_rev:
            skips.append({'id': m['id'], 'reason': 'mapping-ambiguous'})
            continue
        if ok_fwd:
            mapped_ok += 1
            idx = {m['name_a']: 0, m['name_b']: 1}
        elif ok_rev:
            mapped_bad += 1
            idx = {m['name_a']: 1, m['name_b']: 0}
        else:
            skips.append({'id': m['id'], 'reason': 'mapping-mismatch'})
            continue
        g = parse_game(d)
        t = g[idx[target]]
        opp_name = m['name_b'] if m['name_a'] == target else m['name_a']
        o = g[idx[opp_name]]
        t_series, o_series = t.pop('_series'), o.pop('_series')
        anomalies = {'obs0_steps': t.pop('_obs0'), 'obs_multi_steps': t.pop('_obs_multi')}
        o.pop('_obs0'); o.pop('_obs_multi')
        slot = 'a' if m['name_a'] == target else 'b'
        t = dict(t, opp=opp_name, opp_metrics=o,
                 won=(m['winner'] == slot),
                 margin=(sa - sb) if slot == 'a' else (sb - sa),
                 id=m['id'], created=m.get('created'),
                 first=(m['order'][0] == slot),
                 obs_anomalies=anomalies)
        rows.append(t)
        series[m['id']] = {'target': target, 'opp': opp_name,
                           't': t_series, 'o': o_series}
    print(f'parsed {len(rows)} games (map ok={mapped_ok}, swapped={mapped_bad}), '
          f'skipped {len(skips)}')
    for s in skips:
        print(f"  SKIP {s['id']} {s['reason']}")
    if skips:
        print(f'WARNING: {len(skips)} match(es) skipped — see {out_path}.skips.json')
    json.dump(rows, open(out_path, 'w'))
    json.dump(skips, open(out_path + '.skips.json', 'w'))
    with gzip.open(out_path + '.series.json.gz', 'wt') as fh:
        json.dump(series, fh)


if __name__ == '__main__':
    main()
