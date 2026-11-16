#!/usr/bin/env python3
"""Gate-5 cost driver (SYNTHETIC game, seed 777): two pack-level agents play a
full FastCarbon game; every agent() call is timed (wall + process CPU).
Usage: cost_driver.py <pack_path> <tag> <out_json>
The pack forces BLAS threads=1 and the search is single-threaded Python/NumPy,
so per-call CPU time ~= wall time of the same call on a dedicated core inside a
1.5-core quota; wall is also recorded (contention upper bound on a busy host).
"""
import importlib.util, json, sys, time

def load_pack(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def main():
    pack_path, tag, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    mods = [load_pack(f'{tag}_s{s}', pack_path) for s in (0, 1)]
    FC = mods[0]._P['fastsim'].FastCarbon
    sim = FC(); sim.reset(seed=777)
    rows = []
    prev_searched = [0, 0]
    for step in range(300):
        acts = []
        for seat in (0, 1):
            obs = sim.obs(seat)
            w0 = time.perf_counter(); c0 = time.process_time()
            a = mods[seat].agent(obs, None)
            w1 = time.perf_counter(); c1 = time.process_time()
            st = mods[seat].agent.last_stats
            searched_now = 1 if (st.get('searched') or 0) > prev_searched[seat] else 0
            prev_searched[seat] = st.get('searched') or 0
            sims_last = st.get('sims_steps', [[None, 0]])[-1][1] if st.get('sims_steps') else 0
            rows.append({'step': step, 'seat': seat, 'wall': round(w1 - w0, 4),
                         'cpu': round(c1 - c0, 4), 'searched': searched_now, 'sims': sims_last})
            acts.append(a)
        sim.step(acts[0], acts[1])
    json.dump({'tag': tag, 'pack': pack_path, 'rows': rows}, open(out_json, 'w'))
    win = [r for r in rows if r['step'] >= 200]
    import numpy as np
    for name, data in (('all', rows), ('win200', win)):
        w = np.array([r['wall'] for r in data]); c = np.array([r['cpu'] for r in data])
        print(f"{tag} {name}: n={len(data)} wall mean={w.mean():.4f} p95={np.percentile(w, 95):.4f} max={w.max():.4f} | "
              f"cpu mean={c.mean():.4f} p95={np.percentile(c, 95):.4f} max={c.max():.4f} | searched={sum(r['searched'] for r in data)}")

if __name__ == '__main__':
    main()
