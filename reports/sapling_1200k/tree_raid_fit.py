"""Fit the raid intent model on the original by-game split of the Top Five corpus."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
import cnet2_feat as geom
from tree_raid_model import RAID_FEATURES, tree_raid_rows


def metrics(x, y, weights, p0, reach_rates):
    logits = np.clip(x @ weights, -24., 24.)
    p = 1. / (1. + np.exp(-logits))
    return dict(n=len(y), target_actions=int(y.sum()), actual_rate=float(y.mean()),
                predicted=float(p.mean()), brier=float(np.mean((p-y)**2)),
                train_frequency_brier=float(np.mean((p0-y)**2)),
                train_coverage_only_brier=float(np.mean((reach_rates[x[:,4].astype(int)]-y)**2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replays', type=Path, required=True)
    parser.add_argument('--approaches', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    meta_file = HERE / 'top5_h2h_data.json'
    meta = json.loads(meta_file.read_bytes())
    groups = {'train':[], 'val':[]}
    for game in meta['games']:
        path = args.replays / (game['id'] + '.json.gz')
        env = json.loads(gzip.decompress(path.read_bytes()))
        canonical = json.dumps(env, separators=(',', ':'), ensure_ascii=False,
                               allow_nan=False).encode()
        if hashlib.sha256(canonical).hexdigest() != game['canonical_sha256']:
            raise ValueError('replay identity: ' + game['id'])
        rows, labels = [], []
        for i, (frame, after) in enumerate(zip(env['steps'], env['steps'][1:])):
            public = frame[0]['observation']['players']
            for seat in (0, 1):
                # No hidden carbon field or opponent commands enter the predictor.
                obs = dict(players=public, player=seat, step=i)
                actions = after[1-seat]['action'] or {}
                for uid, pos, life, x in tree_raid_rows(obs, geom, env['configuration'], approaches=args.approaches):
                    rows.append(x)
                    origin = public[1-seat][2][uid][0]
                    move = geom.MOVES.index(actions.get(uid) or 'NONE')
                    labels.append(geom.DEST[origin,move] == pos)
        groups[game['split']].append((game['id'], rows, labels))

    def arrays(group):
        return np.array([r for _, rs, _ in group for r in rs], dtype=float), np.array(
            [y for _, _, ys in group for y in ys], dtype=float)
    x, y = arrays(groups['train'])
    constant = float(y.mean())
    reach_rates = np.array([y[x[:,4]==v].mean() for v in [0,1]])
    w = np.zeros(x.shape[1])
    # Convex ridge logistic likelihood. The intercept is unpenalized.
    ridge = np.diag([0.] + [.002]*(x.shape[1]-1))
    for _ in range(25):
        p = 1. / (1. + np.exp(-np.clip(x @ w, -30., 30.)))
        d = np.linalg.solve((x.T * (p*(1-p))) @ x / len(y) + ridge,
                            x.T @ (p-y)/len(y) + ridge @ w)
        w -= d
        if np.max(np.abs(d)) < 1e-8:
            break
    out = dict(approaches=args.approaches, features=RAID_FEATURES, weights=w.tolist(), ridge=.002,
               frontier="conditional on an enemy planter adjacent to or on our live tree, respectively. Label is submitted arrival at that tree, including deaths",
               manifest_sha256=hashlib.sha256(meta_file.read_bytes()).hexdigest(), train_rate=constant, train_rate_by_collector_reach=reach_rates.tolist(), splits={})
    for split, games in groups.items():
        x, y = arrays(games)
        near = x[:, 4] > 0
        out['splits'][split] = dict(games=len(games), total=metrics(x,y,w,constant,reach_rates),
            within_collector_reach=metrics(x[near], y[near], w, constant, reach_rates),
            by_game={gid:metrics(np.array(xs), np.array(ys), w, constant, reach_rates)
                     for gid, xs, ys in games if xs})
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print({k:{j:v for j,v in s.items() if j!='by_game'} for k,s in out['splits'].items()})

if __name__ == '__main__':
    main()
