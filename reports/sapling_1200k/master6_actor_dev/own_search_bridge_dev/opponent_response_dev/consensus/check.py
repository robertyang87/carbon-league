"""Compare two fixed opponent forecasts on original paired terminal branches."""
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / '.git').exists())

def choose(primary, secondary):
    if not all(math.isfinite(v) for v in (primary, secondary)):
        return 'parent'
    return 'alternative' if min(primary, secondary) > 0 else 'parent'

def summarize(rows):
    maps = defaultdict(list)
    for row in rows:
        maps[row['terrain']].append(row)
    return {
        'roots': len(rows), 'maps': len(maps),
        'disagreements': sum(r['old'] != r['calibrated'] for r in rows),
        'changes_vs_old': sum(r['chosen'] != r['old'] for r in rows),
        'changes_vs_parent': sum(r['chosen'] != 'parent' for r in rows),
        'wins': sum(r['scores'][r['chosen']] > 0 for r in rows),
        'map_mean_vs_old': mean(mean(r['scores'][r['chosen']] - r['scores'][r['old']] for r in group) for group in maps.values()),
        'map_mean_vs_parent': mean(mean(r['scores'][r['chosen']] - r['scores']['parent'] for r in group) for group in maps.values()),
    }

def main():
    source = HERE.parent / 'active_calibration/roots.json.gz'
    rows = []
    for item in json.loads(gzip.decompress(source.read_bytes())):
        old, calibrated = (item[k]['h8'] for k in ('old_forecast_delta','new_forecast_delta'))
        chosen = choose(old, calibrated)
        scores = {}
        for arm, result in item['arms'].items():
            raw = (ROOT / item['case'] / ('live_' + arm + '.json')).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == result['receipt_sha256']
            scores[arm] = json.loads(raw)['margin']
        rows.append(dict(case=item['case'], terrain=item['terrain'], batch=item['batch'],
                         old=item['picks']['old_h8'], calibrated=item['picks']['new_h8'],
                         chosen=chosen, scores=scores))
    assert len({r['case'] for r in rows}) == 64
    summary = {name:summarize(group) for name,group in [('earlier', [r for r in rows if r['batch'] <= 6]),
                                                        ('previously_examined_hold', [r for r in rows if r['batch'] > 6])]}
    (HERE/'result.json').write_text(json.dumps(dict(summary=summary, rows=rows),indent=2)+'\n')
    print(summary)

if __name__ == '__main__':
    main()
