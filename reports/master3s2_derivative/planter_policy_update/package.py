"""Rebuild the two own-root planter residuals from the frozen online v2 file."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'ladder/prediction_feedback/package/candidate.py'
PARENT_SHA = '8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'
MODES = ('bc', 'weighted')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(parent, w, b):
    if sha(parent) != PARENT_SHA:
        raise ValueError('Online v2 source hash mismatch')
    arrays = []
    for symbol, value, shape in (('_PLANTER_W', w, (96, 5)), ('_PLANTER_B', b, (5,))):
        if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid coefficient tensor {symbol}')
        data = base64.b64encode(value.astype('<f4').tobytes()).decode()
        arrays.append(f"{symbol} = np.frombuffer(base64.b64decode('{data}'), dtype='<f4').reshape({shape!r})")
    table = '\n'.join(arrays) + '\n\n\n'
    before_return = '''        if planter_cells:
            residual = hf[:, planter_cells].T @ _PLANTER_W + _PLANTER_B
            lw[:, planter_cells] += (np.float32(1.25) * np.tanh(residual / np.float32(1.25))).T
'''
    changes = [
        ('class _Policy:', table + 'class _Policy:'),
        ('    def forward(self, x, sid):', '    def forward(self, x, sid, planter_cells=()):'),
        ('        return lw.T, lb.T', before_return + '        return lw.T, lb.T'),
        ('    lw, lb = _POLICY.forward(x, _SID)',
         "    planter_cells = [w[0] for w in players[me][2].values() if w[2] == 'PLANTER']\n"
         '    lw, lb = _POLICY.forward(x, _SID, planter_cells)'),
    ]
    changes = [(old.replace('\n', '\r\n').encode(), new.replace('\n', '\r\n').encode()) for old, new in changes]
    out = parent
    for old, new in changes:
        if out.count(old) != 1:
            raise ValueError('Ambiguous parent site')
        out = out.replace(old, new)
    undo = out
    for old, new in reversed(changes):
        if undo.count(new) != 1:
            raise ValueError('Ambiguous derived site')
        undo = undo.replace(new, old)
    if undo != parent or len(out) > 10_000_000:
        raise ValueError('Irreversible or oversized package')
    compile(out, '<derived_planter_policy>', 'exec')
    return out


def build_all(head, out_dir):
    parent = PARENT.read_bytes()
    result = {'parent_sha256': sha(parent), 'head_sha256': sha(head.read_bytes()), 'packages': {}}
    with np.load(head, allow_pickle=False) as weights:
        if set(weights.files) != {f'{mode}_{name}' for mode in MODES for name in ('W', 'b')}:
            raise ValueError('Unexpected coefficient keys')
        outputs = {mode:build(parent,weights[mode+'_W'],weights[mode+'_b']) for mode in MODES}
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode, data in outputs.items():
        path = out_dir / (mode + '.py')
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f'Existing {path} has a different identity')
        if not path.exists():
            tmp = path.with_suffix('.tmp'); tmp.write_bytes(data); tmp.replace(path)
        result['packages'][mode] = {'sha256':sha(data),'bytes':len(data),'parameters':485}
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--head', type=Path, default=HERE/'head.npz'); p.add_argument('--out-dir', type=Path, default=HERE)
    a=p.parse_args(); print(json.dumps(build_all(a.head, a.out_dir), indent=2))
