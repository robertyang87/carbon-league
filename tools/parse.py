"""Parse Carbon replays into per-player behavior metrics.

Usage: python3 parse.py <matches.json> <target_name> <replay_dir> <out.json>
Replay files are looked up as <match_id>.json / <match_id>.json.gz / <match_id>-1.json.gz.
Metrics per game for the target player and the opponent:
plants, tree_steps, end_trees, cash@50/100/200/270/final, last30 gain,
recruits, deaths(=recruits-end_workers), peak_cargo, end_cargo, first_plant_step.
Player mapping (replay player0 = name_a) is validated against match scores.
"""
import gzip, json, os, sys


def load_replay(path):
    if path.endswith('.gz'):
        return json.load(gzip.open(path, 'rt'))
    return json.load(open(path))


def find_replay(rdir, mid):
    for cand in (mid + '.json', mid + '.json.gz', mid + '-1.json.gz', mid + '-1.json'):
        p = os.path.join(rdir, cand)
        if os.path.exists(p):
            return p
    return None


def parse_game(d):
    """Return {0: metrics, 1: metrics} from a replay dict."""
    steps = d['steps']
    n = len(steps)
    acc = {p: {'trees_seen': set(), 'workers_seen': set(), 'tree_steps': 0,
               'cash': {}, 'end_trees': 0, 'end_workers': 0, 'peak_cargo': 0.0,
               'end_cargo': 0.0, 'first_plant': None} for p in (0, 1)}
    checkpoints = {50, 100, 200, 269, 270, n - 1}
    for s, step in enumerate(steps):
        snap = None
        for e in step:
            obs = e.get('observation') or {}
            if 'players' in obs:
                snap = obs
                break
        if snap is None:
            continue
        for p in (0, 1):
            pl = snap['players'][p]
            cash, workers, trees = pl[0], pl[2], pl[3]
            a = acc[p]
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
    out = {}
    for p in (0, 1):
        a = acc[p]
        cash = a['cash']
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
        }
    return out


def main():
    matches_path, target, rdir, out_path = sys.argv[1:5]
    matches = json.load(open(matches_path))
    rows, mapped_ok, mapped_bad = [], 0, 0
    for m in matches:
        rp = find_replay(rdir, m['id'])
        if not rp:
            continue
        d = load_replay(rp)
        # validate player0 == name_a via final scores
        ra, rb = d['rewards']
        sa, sb = m['scores']['a'], m['scores']['b']
        if abs(ra - sa) < 1.0 and abs(rb - sb) < 1.0:
            mapped_ok += 1
            idx = {m['name_a']: 0, m['name_b']: 1}
        elif abs(ra - sb) < 1.0 and abs(rb - sa) < 1.0:
            mapped_bad += 1
            idx = {m['name_a']: 1, m['name_b']: 0}
        else:
            continue
        g = parse_game(d)
        t = g[idx[target]]
        opp_name = m['name_b'] if m['name_a'] == target else m['name_a']
        o = g[idx[opp_name]]
        t = dict(t, opp=opp_name, opp_metrics=o, won=(m['winner'] == ('a' if m['name_a'] == target else 'b')),
                 margin=(sa - sb) if m['name_a'] == target else (sb - sa))
        rows.append(t)
    print(f'parsed {len(rows)} games (map ok={mapped_ok}, swapped={mapped_bad})')
    json.dump(rows, open(out_path, 'w'))


if __name__ == '__main__':
    main()
