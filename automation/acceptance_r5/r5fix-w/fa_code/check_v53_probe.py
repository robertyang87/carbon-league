#!/usr/bin/env python3
"""check_v53_probe -- base_v53_probe.py must be behaviour-identical
to base_v53.py (probe insertions are pure reads) and must emit sane
DP-value records.  Mirror games under canonical seeding; final cash
must match exactly, probe records must cover both kinds."""

import importlib.util
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))

import record_trace  # noqa: E402

_ctr = [0]


def load_mod(path):
    _ctr[0] += 1
    name = f"chk_{_ctr[0]}_{os.getpid()}_" + os.path.splitext(
        os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mirror(mod_a, mod_b, seed, side):
    random.seed(7919 * seed + 104729 * side + 31)
    sim = record_trace.FastCarbon(record_trace.CFG).reset(seed=seed)
    agents = [None, None]
    agents[side] = mod_a.agent
    agents[1 - side] = mod_b.agent
    for _ in range(300):
        acts = [agents[p](sim.obs(p), record_trace.CFG) for p in range(2)]
        sim.step(acts[0], acts[1])
    st = sim.get_state()
    return [float(st["cash"][0]), float(st["cash"][1])]


def main():
    ok = True
    for seed, side in ((31000, 0), (31001, 1)):
        probe = load_mod(os.path.join(HERE, "base_v53_probe.py"))
        plain = load_mod(os.path.join(HERE, "base_v53.py"))
        got = mirror(probe, plain, seed, side)
        plain2 = load_mod(os.path.join(HERE, "base_v53.py"))
        plain3 = load_mod(os.path.join(HERE, "base_v53.py"))
        exp = mirror(plain2, plain3, seed, side)
        match = got == exp
        ok &= match
        print(f"seed{seed} side{side}: probe-vs-plain {got} "
              f"plain-vs-plain {exp} -> {'MATCH' if match else 'MISMATCH'}")
        recs = probe.PROBE_RECORDS
        kinds = {}
        vals = []
        for k, _w, v in recs:
            kinds[k] = kinds.get(k, 0) + 1
            vals.append(v)
        import numpy as np
        vals = np.array(vals)
        print(f"  probe records={len(recs)} kinds={kinds} "
              f"value range=[{vals.min():.1f}, {vals.max():.1f}] "
              f"mean={vals.mean():.1f}")
        assert len(recs) > 1000 and kinds.get("C", 0) > 0 \
            and kinds.get("P", 0) > 0
    print("PROBE CHECK", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
