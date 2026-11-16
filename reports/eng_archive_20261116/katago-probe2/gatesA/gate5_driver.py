#!/usr/bin/env python3
"""Part 1 re-registration, pre-field GATE 5 (A form): full-game timing on a
quiet 1.5-core seat. Pack A plays BOTH seats (self-play, worst case for
timing) in a fresh FastCarbon game; every agent() call is timed (wall +
process CPU). The whole game process runs under one systemd-run
CPUQuota=150% scope (see run_gate5.sh) -- per-call wall under the quota is
the 1.5-core-seat reading. Synthetic game seeds only (777+), never the
986200 block.

Usage: gate5_driver.py <seed> <out_json> [max_steps]
"""
import importlib.util, json, sys, time

PACK_A = '/home/arm/junyang/katago_probe_scratch/gatesA/echo3_search_pack.py'


def load_pack(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    seed, out_json = int(sys.argv[1]), sys.argv[2]
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    t0 = time.perf_counter()
    mods = []
    load_times = []
    for s in (0, 1):
        lt0 = time.perf_counter()
        mods.append(load_pack(f'g5_s{seed}_seat{s}', PACK_A))
        load_times.append(round(time.perf_counter() - lt0, 4))
    FC = mods[0]._P['fastsim'].FastCarbon
    sim = FC(); sim.reset(seed=seed)
    rows = []
    prev_searched = [0, 0]
    # Incremental landing (owner rule 2026-11-15): every call appends one
    # JSONL row immediately, so progress = line count of the rows file,
    # readable externally at any moment; the summary JSON lands at game end.
    rows_path = out_json.replace('.json', '_rows.jsonl')
    rf = open(rows_path, 'w')
    game_w0 = time.perf_counter(); game_c0 = time.process_time()
    for step in range(max_steps):
        acts = []
        for seat in (0, 1):
            obs = sim.obs(seat)
            w0 = time.perf_counter(); c0 = time.process_time()
            a = mods[seat].agent(obs, None)
            w1 = time.perf_counter(); c1 = time.process_time()
            st = mods[seat].agent.last_stats
            searched_now = 1 if (st.get('searched') or 0) > prev_searched[seat] else 0
            prev_searched[seat] = st.get('searched') or 0
            row = {'step': step, 'seat': seat,
                   'wall': round(w1 - w0, 4), 'cpu': round(c1 - c0, 4),
                   'searched': searched_now}
            rows.append(row)
            rf.write(json.dumps(row) + '\n'); rf.flush()
            acts.append(a)
        sim.step(acts[0], acts[1])
    rf.close()
    out = {'seed': seed, 'pack': PACK_A, 'max_steps': max_steps,
           'module_load_s': load_times,
           'first_call_wall': [next(r['wall'] for r in rows if r['seat'] == s) for s in (0, 1)],
           'game_wall_s': round(time.perf_counter() - game_w0, 3),
           'game_cpu_s': round(time.process_time() - game_c0, 3),
           'proc_total_s': round(time.perf_counter() - t0, 3),
           'rows': rows}
    json.dump(out, open(out_json, 'w'))
    print(f'gate5 seed {seed} DONE wall={out["game_wall_s"]}', flush=True)


if __name__ == '__main__':
    main()
