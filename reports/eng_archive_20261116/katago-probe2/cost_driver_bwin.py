#!/usr/bin/env python3
"""Gate-5 window insurance run (mirrors cost_driver.py harness exactly):
steps [0,200) played by SEARCH=False agents (no-search path, untouched by
the sims edits), steps [200,300) by pack B stock-CONF agents (provisional
sims settings), per-call wall+CPU recorded for the window only.
Caveat: the B agents start cold at step 200 (fresh make_search_agent
instances); a sensitivity line excluding the first 10 window calls is
printed alongside. Independent seed 777001 (reset seed only).
"""
import base64, importlib.util, json, os, time, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B_PATH = os.path.join(HERE, 'echo3_sims_pack.py')
SEED = 777001

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def weights(m):
    w = m._dequantize(zlib.decompress(base64.b64decode(m._B1)), m._Q1)
    w['meta'] = np.array(m._M1)
    return w

def main():
    mB = load(B_PATH, 'pack_B_costw')
    mL = load(B_PATH, 'pack_lead_costw')
    wB, wL = weights(mB), weights(mL)
    FC = mB._P['fastsim'].FastCarbon
    sim = FC(); sim.reset(seed=SEED)
    lead = [mL._S.make_search_agent(dict(mL.CONF, SEARCH=False), weights=wL,
                                    rweights=None, oweights=None) for _ in (0, 1)]
    win = [mB._S.make_search_agent(dict(mB.CONF), weights=wB,
                                   rweights=None, oweights=None) for _ in (0, 1)]
    recs = []
    for step in range(300):
        acts = []
        for seat in (0, 1):
            obs = sim.obs(seat)
            if step >= 200:
                t0 = time.perf_counter(); c0 = time.process_time()
                a = win[seat](obs)
                rec = {'step': step, 'seat': seat,
                       'wall': round(time.perf_counter() - t0, 4),
                       'cpu': round(time.process_time() - c0, 4),
                       'searched': int(win[seat].stats.get('searched', 0)),
                       'depth': int(win[seat].stats.get('depth_sum', 0))}
                recs.append(rec)
                if len(recs) % 20 == 0:
                    print(f'win calls={len(recs)} step={step}', flush=True)
            else:
                a = lead[seat](obs)
            acts.append(a)
        sim.step(acts[0], acts[1])
    json.dump({'tag': 'Bwin', 'seed': SEED, 'records': recs},
              open(os.path.join(HERE, 'cost_Bwin.json'), 'w'))
    arr = np.array([r['cpu'] for r in recs])
    arr2 = np.array([r['cpu'] for r in recs[10:]])
    print(f"Bwin window: n={len(recs)} cpu mean={arr.mean():.4f} p95={np.percentile(arr, 95):.4f} max={arr.max():.4f}", flush=True)
    print(f"Bwin window excl-first-10: n={len(arr2)} cpu mean={arr2.mean():.4f} p95={np.percentile(arr2, 95):.4f} max={arr2.max():.4f}", flush=True)

if __name__ == '__main__':
    main()
