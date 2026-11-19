"""Fit a one-turn enemy move distribution on frozen Top Five self-play IDs."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RULES = ROOT / 'reports/master_family/tactical_rules'
sys.path.insert(0, str(RULES))
import opponent_intent as intent
from top5_intent_check import own_view


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_game(folder, item):
    raw = (folder / (item['id'] + '.json.gz')).read_bytes()
    obj = json.loads(gzip.decompress(raw))
    if hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() != item['git_blob']:
        canon = json.dumps(obj, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        if sha(canon) != item['canonical_sha256']:
            raise ValueError('wrong game ' + item['id'])
    return obj


def select(items, split, n):
    return sorted((v for v in items if v['split'] == split),
                  key=lambda v: sha(('sapling-cargo-interception-v1/' + v['id']).encode()))[:n]


def examples(items, folder):
    xs, ys, roles, close = [], [], [], []
    for item in items:
        steps = read_game(folder, item)['steps']
        for t in range(item['seed'] % 13, len(steps) - 1, 13):
            for seat in (0, 1):
                now = own_view(steps[t], seat)
                previous = own_view(steps[t-1], seat) if t else None
                ids, rk, x = intent.features(now, previous)
                if not ids:
                    continue
                labels = intent.recorded_moves(steps[t+1][1-seat]['action'] or {}, ids)
                mine = [w[0] for w in now['players'][seat][2].values() if w[2] == 'COLLECTOR']
                foes = now['players'][1-seat][2]
                close.extend([bool(mine) and intent.DIST[foes[uid][0],mine].min() <= 2 for uid in ids])
                xs.append(x); ys.extend(labels); roles.extend(rk)
    return np.concatenate(xs), np.asarray(ys), np.asarray(roles), np.asarray(close)


def score(dataset, weights, prior):
    x,y,roles,close = dataset
    ps = intent.probabilities(x,roles,weights)
    out = {}
    for name,mask in [('all',np.ones(len(y),bool)),('two_steps_of_collector',close)]:
        v,ref,k = ps[mask], prior[roles[mask]], y[mask]
        out[name] = dict(n=len(k), logloss=float(-np.log(v[np.arange(len(k)),k]).mean()),
                         frequency_logloss=float(-np.log(ref[np.arange(len(k)),k]).mean()),
                         top1=float((v.argmax(1)==k).mean()))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--replays',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    manifest=json.loads(a.manifest.read_text())
    groups={name:select(manifest['games'],name,n) for name,n in [('train',32),('val',16)]}
    train=examples(groups['train'],a.replays)
    weights,prior=intent.fit((None,train[2],train[0],train[1]))
    valid=examples(groups['val'],a.replays)
    obj=dict(schema=1,features=intent.FEATURES,weights=weights.tolist(),role_frequency=prior.tolist(),
             train=score(train,weights,prior),validation=score(valid,weights,prior),
             game_ids={k:[g['id'] for g in v] for k,v in groups.items()},
             source_sha256={p.name:sha(p.read_bytes()) for p in (Path(__file__),RULES/'opponent_intent.py',a.manifest)})
    a.out.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
