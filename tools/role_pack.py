"""Build a worker-role data pack from parse2 series output.

Data only — no verdicts. Groups: target-in-wins, target-in-losses,
opponent-in-target-wins, opponent-in-target-losses.
Usage: python3 role_pack.py <parsed_v2.json> <out_pack.json>
The series file is <parsed_v2.json>.series.json.gz next to the parsed file.
"""
import gzip, json, statistics as st, sys

CKPT = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 299]
PHASES = {'p1_0_99': (0, 100), 'p2_100_199': (100, 200), 'p3_200_299': (200, 300)}
SERIES_KEYS = ['cash', 'trees', 'workers', 'cargo', 'planters', 'collectors']


def mean_series(list_of_series, key):
    out = []
    for s in range(300):
        vals = [d[key][s] for d in list_of_series if d[key][s] is not None]
        out.append(round(st.mean(vals), 3) if vals else None)
    return out


def phase_mean(series, key, lo, hi):
    vals = [x for x in series[key][lo:hi] if x is not None]
    return round(st.mean(vals), 3) if vals else None


def first_ge(series, key, k):
    for i, x in enumerate(series[key]):
        if x is not None and x >= k:
            return i
    return None


def main():
    parsed_path, out_path = sys.argv[1:3]
    rows = json.load(open(parsed_path))
    series = json.load(gzip.open(parsed_path + '.series.json.gz', 'rt'))
    groups = {'target_win': [], 'target_loss': [], 'opp_win': [], 'opp_loss': []}
    per_game = []
    for r in rows:
        s = series[r['id']]
        t, o = s['t'], s['o']
        g = 'win' if r['won'] else 'loss'
        groups[f'target_{g}'].append(t)
        groups[f'opp_{g}'].append(o)
        tp_share = [p / (p + c) if (p + c) > 0 else None
                    for p, c in zip(t['planters'], t['collectors'])]
        share_vals = [x for x in tp_share if x is not None]
        per_game.append({
            'id': r['id'], 'created': r['created'], 'opp': r['opp'],
            'won': r['won'], 'margin': r['margin'], 'first': r['first'],
            'planters_mean': round(st.mean([x for x in t['planters'] if x is not None]), 3),
            'planters_p1': phase_mean(t, 'planters', 0, 100),
            'planters_p2': phase_mean(t, 'planters', 100, 200),
            'planters_p3': phase_mean(t, 'planters', 200, 300),
            'collectors_mean': round(st.mean([x for x in t['collectors'] if x is not None]), 3),
            'planter_share': round(st.mean(share_vals), 4) if share_vals else None,
            'max_planters': max(x for x in t['planters'] if x is not None),
            'first_step_planters_ge2': first_ge(t, 'planters', 2),
            'first_step_planters_ge3': first_ge(t, 'planters', 3),
            'workers_50': t['workers'][50], 'workers_100': t['workers'][100],
            'workers_200': t['workers'][200], 'workers_299': t['workers'][299],
            'trees_100': t['trees'][100], 'trees_150': t['trees'][150],
            'trees_200': t['trees'][200], 'trees_250': t['trees'][250],
            'cash_100': t['cash'][100], 'cash_200': t['cash'][200],
            'cash_270': t['cash'][270],
            'peak_cargo': r['peak_cargo'],
            'peak_cargo_step': max(range(300), key=lambda i: (t['cargo'][i] is not None, t['cargo'][i])),
            'end_trees': r['end_trees'], 'end_cargo': r['end_cargo'],
            'recruits': r['recruits'],
            'opp_planters_mean': round(st.mean([x for x in o['planters'] if x is not None]), 3),
            'opp_planters_p1': phase_mean(o, 'planters', 0, 100),
            'opp_planters_p2': phase_mean(o, 'planters', 100, 200),
            'opp_planters_p3': phase_mean(o, 'planters', 200, 300),
            'opp_collectors_mean': round(st.mean([x for x in o['collectors'] if x is not None]), 3),
            'opp_trees_150': o['trees'][150],
            'opp_peak_cargo_step': max(range(300), key=lambda i: (o['cargo'][i] is not None, o['cargo'][i])),
        })
    pack = {
        'meta': {
            'source': parsed_path,
            'n_games': len(rows),
            'n_wins': sum(1 for r in rows if r['won']),
            'n_losses': sum(1 for r in rows if not r['won']),
            'definitions': {
                'role': "worker entry index-2 string; corpus contains exactly 'PLANTER' and 'COLLECTOR', so planters+collectors = workers at every step",
                'groups': 'target_win/loss = the parsed target in its wins/losses; opp_win = opponents in games the target won; opp_loss = opponents in games the target lost',
                'series': 'per-step values from the single observation event of each step (300 steps/game, no missing steps in these corpora)',
                'planter_share': 'per-step planters/(planters+collectors), averaged over steps with at least one worker',
            },
        },
        'step_means': {g: {k: mean_series(ds, k) for k in SERIES_KEYS}
                       for g, ds in groups.items()},
        'checkpoints': {g: {str(c): {k: round(st.mean([d[k][c] for d in ds if d[k][c] is not None]), 3)
                                     for k in SERIES_KEYS} for c in CKPT}
                        for g, ds in groups.items()},
        'phase_means': {g: {ph: {'planters': round(st.mean([phase_mean(d, 'planters', lo, hi) for d in ds]), 3),
                                 'collectors': round(st.mean([phase_mean(d, 'collectors', lo, hi) for d in ds]), 3),
                                 'workers': round(st.mean([phase_mean(d, 'workers', lo, hi) for d in ds]), 3),
                                 'trees': round(st.mean([phase_mean(d, 'trees', lo, hi) for d in ds]), 3),
                                 'cargo': round(st.mean([phase_mean(d, 'cargo', lo, hi) for d in ds]), 3)}
                            for ph, (lo, hi) in PHASES.items()}
                        for g, ds in groups.items()},
        'per_game': per_game,
    }
    json.dump(pack, open(out_path, 'w'))
    print(f'wrote {out_path}: {len(rows)} games '
          f'({pack["meta"]["n_wins"]}W/{pack["meta"]["n_losses"]}L)')


if __name__ == '__main__':
    main()
