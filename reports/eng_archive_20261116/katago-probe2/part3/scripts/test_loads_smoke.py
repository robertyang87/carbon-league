#!/usr/bin/env python3
"""Self-test 1: py_compile + import + factory construction for orig / base-A /
arm A / arm B packs, shipped-CONF single-feed smoke on SYNTHETIC roots
(tasks8, SYNTH-SEED777), plus a DIAGNOSTIC relaxed-budget run (BUDGET=10,
HARD=30 -- test-only conf override, NOT the shipped bytes' behavior) proving
search engagement end-to-end: on this throttled VM the shipped wall-clock
BUDGET=0.7 exits shallow before any leaf call (Part 1 receipt S4 artifact),
so engagement is read from searched/shallow/depth stats under the relaxed
budget. Smoke only -- NOT a construction gate."""
import base64, importlib.util, json, py_compile, sys, zlib
import numpy as np

ROOTS = [json.loads(l) for l in open('/home/hatch/workspace/katago-probe/impl/smoke/tasks8.jsonl')]
PATHS = {
    'orig': '/home/hatch/workspace/katago-probe/impl/echo3_ref.py',
    'baseA': '/home/hatch/workspace/katago-probe2/modules/echo3_search_pack.py',
    'A': '/home/hatch/workspace/katago-probe2/part3/modules/echo3_part3_a_pack.py',
    'B': '/home/hatch/workspace/katago-probe2/part3/modules/echo3_part3_b_pack.py',
}

def load_pack(name, path):
    py_compile.compile(path, doraise=True)
    spec = importlib.util.spec_from_file_location(f'pack3_{name}', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    w1 = m._dequantize(zlib.decompress(base64.b64decode(m._B1)), m._Q1)
    w1['meta'] = np.array(m._M1)
    return m, w1

def run_roots(m, w, roots, conf=None):
    out = []
    for idx, r in enumerate(roots):
        np.random.seed(20261114 + idx)
        ag = m._S.make_search_agent(conf if conf is not None else m.CONF, weights=w)
        a = ag(r['obs'])
        st = ag.stats
        out.append({'idx': idx, 'seat': r['seat'], 'step': r['step'], 'actions': a,
                    'searched': st.get('searched', 0), 'errors': st.get('errors', 0),
                    'shallow': st.get('shallow', 0), 'depth_sum': st.get('depth_sum', 0),
                    'branches': st.get('branches', 0), 'changed': st.get('changed', 0)})
    return out

mods, ws, res = {}, {}, {}
for name, path in PATHS.items():
    mods[name], ws[name] = load_pack(name, path)
    res[name] = run_roots(mods[name], ws[name], ROOTS)
    tot = {k: sum(r[k] for r in res[name]) for k in ('searched', 'errors', 'shallow')}
    print(f"{name}: py_compile+import+factory+feed OK; roots={len(ROOTS)} totals={tot}")
errs = sum(r['errors'] for n in PATHS for r in res[n])
assert errs == 0, f'errors: {errs}'

def changed(x, y, src=res):
    return sum(1 for a, b in zip(src[x], src[y]) if a['actions'] != b['actions'])
print('shipped-conf changed A vs orig:', changed('A', 'orig'), '/8; B vs orig:', changed('B', 'orig'),
      '/8 (synthetic; local wall-clock shallow artifact expected)')

diag = {}
for arm in ('A', 'B'):
    conf = dict(mods[arm].CONF, BUDGET=10.0, HARD=30.0)
    diag[arm] = run_roots(mods[arm], ws[arm], ROOTS, conf=conf)
    reached = sum(1 for r in diag[arm] if r['searched'] and r['shallow'] < r['searched'])
    tot = {k: sum(r[k] for r in diag[arm]) for k in ('searched', 'shallow', 'depth_sum', 'branches', 'changed', 'errors')}
    print(f"DIAG {arm} (BUDGET=10/HARD=30): totals={tot}; roots reaching leaf evaluation: {reached}/8")
    assert tot['errors'] == 0 and reached >= 1, f'diagnostic engagement failed for arm {arm}'
print('DIAG changed A vs B:', changed('A', 'B', diag), '/8 (diagnostic budget, synthetic roots)')
json.dump({'shipped': {n: [{'idx': r['idx'], 'searched': r['searched'], 'shallow': r['shallow']} for r in res[n]] for n in res},
           'diag': {n: [{'idx': r['idx'], 'searched': r['searched'], 'shallow': r['shallow'],
                         'depth_sum': r['depth_sum'], 'branches': r['branches'], 'changed': r['changed']} for r in diag[n]] for n in diag}},
          open('/home/hatch/workspace/katago-probe2/part3/data/smoke_results.json', 'w'), indent=1)
print('SMOKE OK -> data/smoke_results.json')
