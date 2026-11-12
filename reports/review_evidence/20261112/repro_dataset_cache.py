"""Build a 3-sample synthetic replay twice, changing config; outputs live only in /tmp."""
import gzip
import json
import os
import pathlib
import sys
import tempfile

os.environ['BUILD_WORKERS'] = '1'
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'code'))
import numpy as np
import cnet2_build as B
from fastsim import FastCarbon
import eval_v53

with tempfile.TemporaryDirectory(prefix='carbon-bc-cache-review-') as tmp:
    root = pathlib.Path(tmp)
    rd = root / 'corpus'
    out = root / 'dataset'
    (rd / 'replays').mkdir(parents=True)
    sim = FastCarbon(eval_v53.CFG).reset(seed=9123)
    steps = []
    for t in range(4):
        if t:
            sim.step({}, {})
        steps.append([{'observation': sim.obs(p), 'status': 'ACTIVE', 'action': {}}
                      for p in (0, 1)])
    with gzip.open(rd / 'replays/demo-0.json.gz', 'wt') as f:
        json.dump({'steps': steps}, f)
    # Builder normally filters by index length. Declared 300 lets this isolated
    # 3-decision fixture exercise shard reuse without running a whole match.
    with open(rd / 'index.json', 'w') as f:
        json.dump([{'file': 'replays/demo-0.json.gz', 'steps': 300, 'order': [0, 1],
                    'strategies': [{'name': B.F.TEACHERS[0]}, {'name': 'not-a-teacher'}],
                    'scores': [100, 0]}], f)
    B.main(str(out), [str(rd)])
    print(json.dumps({'first_build_steps': np.load(out / 'step.npy').tolist(),
                      'first_build_teachers': np.load(out / 'teacher.npy').tolist()}))
    B.STEP_MIN = 2
    B.F.TEACHERS[0], B.F.TEACHERS[1] = B.F.TEACHERS[1], B.F.TEACHERS[0]
    B.main(str(out), [str(rd)])
    actual_steps = np.load(out / 'step.npy').tolist()
    actual_teachers = np.load(out / 'teacher.npy').tolist()
    print(json.dumps({'new_config_STEP_MIN': 2, 'teacher_rows_swapped': [0, 1],
                      'actual_steps': actual_steps, 'actual_teachers': actual_teachers,
                      'expected_steps': [2], 'expected_teachers': [1],
                      'metadata_row0_now': json.loads((out / 'meta.json').read_text())['teachers'][0]}))
    assert actual_steps == [0, 1, 2] and actual_teachers == [0, 0, 0]
