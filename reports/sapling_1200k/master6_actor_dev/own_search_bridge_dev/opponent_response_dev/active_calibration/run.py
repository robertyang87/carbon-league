"""Fit one NONE-logit offset against original command frequencies on six training maps."""
from pathlib import Path
import sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import train
import transfer

def main():
    b = train.bridge
    source = HERE.parent / 'opponent.npz'
    w = dict(np.load(source, allow_pickle=False))
    records, _ = train.identity()
    gaps = []
    active = 0
    for record in records:
        if record['split'] != 'fit':
            continue
        for _, _, enemy, x, actions in train.source_frames(record):
            labels = enemy.labels(actions)
            for slot, matrix, legal in train.training_inputs(enemy, x, w):
                if not (legal[0] and legal[1:].any() and legal[labels[slot]]):
                    continue
                scores = b.predict(w, matrix)
                gaps.append(float(max(scores[1:][legal[1:]]) - scores[0]))
                active += int(labels[slot] != 0)
    delta = float(np.quantile(gaps, 1 - active / len(gaps)))
    # The first action identity column is one for NONE and zero for every other action.
    extra = np.zeros((len(w['hidden']), 1), dtype=np.float32)
    extra[-6, 0] = 1
    calibrated = dict(hidden=np.column_stack([w['hidden'], extra]),
                      bias=np.append(w['bias'], np.float32(0)),
                      head=np.vstack([w['head'], np.float32(delta / np.tanh(1.))]))
    assert np.allclose(b.predict(calibrated, matrix) - b.predict(w, matrix),
                       [delta, 0, 0, 0, 0], atol=3e-5)
    np.savez_compressed(HERE / 'opponent.npz', **calibrated)
    train.HERE = HERE
    result = train.evaluate(records, w, calibrated)
    b.save(HERE / 'evaluation.json', result)
    b.save(HERE / 'fit.json', dict(train_decisions=len(gaps), active=active, delta=delta,
                                   parent_sha256=b.sha(source), weights_sha256=b.sha(HERE/'opponent.npz')))
    rows = transfer.audit_roots(calibrated)
    transfer.verify_disjoint(rows, {r['terrain'] for r in records})
    b.save(HERE / 'roots.json.gz', rows)
    b.save(HERE / 'terminal.json', {k: transfer.summarize(group) for k, group in
           [('all', rows), ('historical_hold', [r for r in rows if r['batch'] > 6])]})

if __name__ == '__main__':
    main()
