"""Read the old tree-return model at every real planting in two games against the same entrant."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LADDER = HERE.parent
FAMILY = LADDER.parent
sys.path.insert(0, str(FAMILY.parents[1] / 'reports/sapling_1200k/all_top5'))
from context import Game

spec = importlib.util.spec_from_file_location('tree_public_features', LADDER / 'tree_payback/runtime.py')
feat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feat)
MODEL = FAMILY / 'tree_control_response/weights.json'
RISK = LADDER / 'tree_payback/weights.json'
GAMES = [('v2', 'nn_patch_cutoff', 'ffb3d312918eb0c4d49a9912ce4f77f7', 0, 'v2_'),
         ('v4', 'v4_observations', '1c80b916b203238a43267baeabc88fbe', 1, '')]


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(frame, hist, cfg, item, model, risk):
    uid = item['planter']
    seat = frame['player']
    assert frame['players'][seat][2][uid][0] == item['cell'] and 'full_carbon' not in frame
    x = feat.features(frame, uid, cfg, hist)
    quoted = feat.quoted_price(frame['players'], seat, cfg)
    assert abs(quoted - item['cost']) < 1e-7
    z = [(a - b) / c for a,b,c in zip(x, model['feature_mean'], model['feature_scale'], strict=True)]
    net, hold = [mu + sum(a * w[j] for a, w in zip(z, model['coef'], strict=True))
                 for j, mu in enumerate(model['means'])]
    return dict(step=frame['step'], planter=uid, cell=item['cell'], tree=item['tree'], cost=quoted,
                nearest_enemy_planter_tiles=round(x[3] * cfg['size']), predicted_loss_probability=feat.predict(x,risk),
                predicted_net=net, predicted_tenure=hold, actual_net=item['net'],
                actual_duration=item['end_t']-item['t'], exit=item['exit'])


def totals(rows):
    flagged = [r for r in rows if r['predicted_net'] < 0]
    true_positives = [r for r in flagged if r['actual_net'] < 0]
    false_positives = [r for r in flagged if r['actual_net'] >= 0]
    missed = [r for r in rows if r['actual_net'] < 0 and r['predicted_net'] >= 0]
    return dict(plants=len(rows), actual_losses=sum(r['actual_net'] < 0 for r in rows),
                predicted_loss=len(flagged), losses_flagged=len(true_positives), profitable_flagged=len(false_positives),
                actual_losses_missed=len(missed),
                first_tenure_loss_on_flagged=sum(-r['actual_net'] for r in true_positives),
                first_tenure_profit_on_flagged=sum(r['actual_net'] for r in false_positives),
                first_tenure_loss_on_missed=sum(-r['actual_net'] for r in missed))


def main():
    fit = read(MODEL)
    risk = read(RISK)['full']
    results = []
    for version, directory, match, seat, prefix in GAMES:
        replay_path = LADDER / directory / 'replays' / (match+'-1.json.gz')
        ledger_path = LADDER / directory / (prefix+match+'-1.ledger.json.gz')
        replay, ledger = read(replay_path), read(ledger_path)
        g = Game(replay)
        teams = []
        for index, planting in enumerate(ledger['planting']):
            s = seat if index == 0 else 1-seat
            rows = []
            for item in planting['individual']:
                t = item['t']
                assert item['seat'] == s
                frame = g._obs(t,s)
                hist = {d:g._obs(t-d,s)['players'] for d in (3,5) if t>=d}
                rows.append(evaluate(frame,hist,replay['configuration'],item,fit,risk))
            teams.append(dict(side=version if index == 0 else 'RyanX', seat=s, summary=totals(rows), rows=rows))
        results.append(dict(match=match, replay_sha256=sha(replay_path), ledger_sha256=sha(ledger_path),teams=teams))
    record = dict(model_sha256=sha(MODEL), risk_sha256=sha(RISK), cutoff='predicted net income < 0',
                  population='All trees actually planted, first uninterrupted tenure per tree. No simulated decisions.',
                  games=results)
    (HERE/'tree_forecast.json').write_text(json.dumps(record,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    for g in results:
        print([(t['side'],t['summary']) for t in g['teams']])


if __name__ == '__main__':
    main()
