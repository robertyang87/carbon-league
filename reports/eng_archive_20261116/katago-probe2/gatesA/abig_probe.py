#!/usr/bin/env python3
"""Confirmation probe for the gate-2 cold-start budget mechanism.

Takes N searched roots from shard 8's landed output (deterministic: the
first N rows with A_searched>0, in file order) and re-runs, per root and
with the same per-root seed:
  orig      : original pack, stock conf (default move reference)
  A_stock   : pack A, stock conf (control -- must reproduce shard 8:
              shallow, depth 1, unchanged)
  A_big     : pack A, harness-only override BUDGET=30 / HARD=90
              (run_boundary.py precedent; pack bytes untouched)
If A_big reaches depth >= MIN_DEPTH and changes some moves while A_stock
reproduces shallow/unchanged, the mechanism is confirmed: the sweep is
live but the native wall/CPU budget is consumed by cold-start phase-1
before the depth loop starts (agent() sets t0/tw at entry; the loop's
first break fires at 0.45*BUDGET with no depth completed).

Usage: abig_probe.py <shard8_json> <n_roots> <out_json>
"""
import json, sys
import numpy as np
import gate2_driver as G


def main():
    shard8_path, n_want, out_json = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    rows8 = json.load(open(shard8_path))['rows']
    picked = [r for r in rows8 if r['A_searched'] > 0][:n_want]
    idxs = {r['root_index'] for r in picked}
    tasks = {t['root_index']: t for t in G.tasks_from_replays(G.ROOTS_DIR, G.ROOTSET)}
    mo, wo = G.load_pack('pack_orig_ab', G.PACK_O)
    mA, wA = G.load_pack('pack_A_ab', G.PACK_A)
    conf_o = dict(mo.CONF)
    conf_A = dict(mA.CONF)
    conf_big = dict(mA.CONF); conf_big['BUDGET'] = 30.0; conf_big['HARD'] = 90.0
    out = []
    for r in picked:
        t = tasks[r['root_index']]
        seed = 20261114 + t['root_index']
        ao, _s, eo, _w, _c = G.call(mo, wo, conf_o, t['obs'], seed)
        aA, sA, eA, _w2, _c2 = G.call(mA, wA, conf_A, t['obs'], seed)
        aB, sB, eB, _w3, _c3 = G.call(mA, wA, conf_big, t['obs'], seed)
        jo = json.dumps(ao, sort_keys=True) if ao is not None else None
        jA = json.dumps(aA, sort_keys=True) if aA is not None else None
        jB = json.dumps(aB, sort_keys=True) if aB is not None else None
        rec = {'root_index': t['root_index'], 'step': t['step'], 'seat': t['seat'],
               'shard8_changed': r['changed'], 'shard8_depth': r['A_depth'],
               'A_stock_searched': int(sA.get('searched', 0)),
               'A_stock_shallow': int(sA.get('shallow', 0)),
               'A_stock_depth': int(sA.get('depth_sum', 0)),
               'A_stock_changed': (jo is not None and jA is not None and jo != jA),
               'A_big_searched': int(sB.get('searched', 0)),
               'A_big_shallow': int(sB.get('shallow', 0)),
               'A_big_depth': int(sB.get('depth_sum', 0)),
               'A_big_changed': (jo is not None and jB is not None and jo != jB),
               'exc': [eo, eA, eB]}
        out.append(rec)
        print(rec, flush=True)
    json.dump({'rows': out}, open(out_json, 'w'))
    print('ABIG_PROBE_DONE', flush=True)


if __name__ == '__main__':
    main()
