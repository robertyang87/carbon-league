"""Verify the alleged separate rollout and main networks on frozen original inputs."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def main(source):
    base = Path(__file__).resolve().parent
    orig = Path(source).read_bytes()
    expected = '65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
    assert hashlib.sha256(orig).hexdigest() == expected
    modspec = importlib.util.spec_from_file_location('native_original', source)
    model = importlib.util.module_from_spec(modspec)
    modspec.loader.exec_module(model)
    model._torch.set_num_threads(1)
    net = model._net()
    assert model._RNET is net and not hasattr(model,'_ROLL')
    panel = base.parent / 'rollout_readout/timing/panel.npz'
    spec = json.loads((panel.parent / 'panel.json').read_text())
    results = []
    with np.load(panel) as archive:
        rows = spec['public_roots']
        for root in rows[:4]:
            selected = next(b for b in root['batches'] if b['kind'] == 'joint')
            x = archive[selected['key']].astype(np.float32)*np.float32(.01)
            sid = selected['sid']
            w, b, _ = net(x,sid)
            w2, b2, _ = model._RNET(x,sid)
            assert np.array_equal(w,w2) and np.array_equal(b,b2)
            results.append(dict(map_seed=root['seed'],seat=root['seat'],step=root['t'],rows=len(x),
                                worker_max_abs=float(np.abs(w-w2).max()),base_max_abs=float(np.abs(b-b2).max())))
    stores = sorted({a.id for a in ast.walk(ast.parse(orig)) if isinstance(a,ast.Name) and isinstance(a.ctx,ast.Store) and a.id=='_ROLL'})
    result=dict(original=expected,roll_artifact_assignments=stores,instance_identity=model._RNET is net,
                backend=model._S_STATS['rnet'],roll_temperature=model._S_RTEMP,panel_sha256=hashlib.sha256(panel.read_bytes()).hexdigest(),checks=results)
    (base/'identity.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main(sys.argv[1])
