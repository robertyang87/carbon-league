"""Time the readout change while hashing the previously archived real search inputs."""
import argparse
import json
import os
from time import perf_counter_ns, process_time_ns
import types

from build import FAMILY, HERE, PARENT, PARENT_SHA, build, sha

MEASUREMENTS = FAMILY / 'rollout_readout/timing'


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--cpu', required=True, type=int)
    args = cli.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['CARBON_DEBUG'] = '1'
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    raw = (MEASUREMENTS / 'panel.json').read_bytes()
    panel = json.loads(raw)
    features_file = MEASUREMENTS / 'panel.npz'
    assert sha(features_file.read_bytes()) == panel['features_sha256']
    packages = (PARENT_SHA, build())
    modules = []
    for path, digest in zip((PARENT, HERE / 'candidate.py'), packages, strict=True):
        package = path.read_bytes()
        assert sha(package) == digest
        mod = types.ModuleType('head_measurement')
        exec(compile(package, str(path), 'exec'), mod.__dict__)
        modules.append(mod)
    torch.set_num_threads(1)
    assert all(type(m._net()).__name__ == '_WinoNet' for m in modules)
    captures = []
    with np.load(features_file, allow_pickle=False) as arrays:
        for case in panel['public_roots']:
            inputs = []
            for index, batch in enumerate(case['batches']):
                key = batch['key']
                x = arrays[key].astype(np.float32) * np.float32(.01)
                sc = arrays[key+'s'] if batch['scalar'] else None
                stride = 2 if batch['kind'] == 'joint' else 1
                terminal = batch['kind'] == 'own' and case['depth'] == 8 and index == len(case['batches'])-1
                assert stride == 1 or sc is not None and len(x) % 2 == 0
                inputs.append((x, batch['sid'], sc, stride, terminal))
            captures.append(inputs)
    rows = []
    for i, inputs in enumerate(captures):
        orders = ([0, 1], [1, 0], [0, 1]) if i % 2 == 0 else ([1, 0], [0, 1], [1, 0])
        timings = [[], []]
        for order in orders:
            outputs = [None, None]
            for version in order:
                net = modules[version]._NET
                cpu, wall = process_time_ns(), perf_counter_ns()
                if version == 0:
                    actual = [net(x, sid, sc) for x, sid, sc, stride, terminal in inputs]
                else:
                    actual = [net(x, sid, sc, value_stride=stride, policy=not terminal)
                              for x, sid, sc, stride, terminal in inputs]
                timings[version].append([process_time_ns()-cpu, perf_counter_ns()-wall])
                outputs[version] = actual
            checks = []
            for (_, _, _, stride, terminal), a, b in zip(inputs, *outputs, strict=True):
                if terminal:
                    assert b[:2] == (None, None)
                else:
                    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                assert (a[2] is None) == (b[2] is None)
                delta = 0 if a[2] is None else float(np.max(np.abs(a[2][::stride]-b[2])))
                checks.append(delta)
            assert max(checks) <= 2e-6
        rows.append(dict(seed=panel['public_roots'][i]['seed'], batch_count=len(inputs), times_ns=timings,
                         unique_max_abs_value_delta=max(checks), opponent_values_removed=sum(len(x)//2 for x,_,_,w,_ in inputs if w==2),
                         policy_rows_removed=sum(len(x) for x,_,_,_,terminal in inputs if terminal)))
    (HERE / 'inference.json').write_text(json.dumps(dict(packages=packages, panel=sha(raw), features=panel['features_sha256'],
                                                       torch=str(torch.__version__), numpy=np.__version__, cpu=args.cpu,
                                                       rows=rows), allow_nan=False, indent=2)+'\n')
    print('timed', sum(r['batch_count'] for r in rows), 'batches; exact retained policies; max value delta',
          max(r['unique_max_abs_value_delta'] for r in rows))


if __name__ == '__main__':
    main()
