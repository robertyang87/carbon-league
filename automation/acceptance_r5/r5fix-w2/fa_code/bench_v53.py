"""bench_v53: import cost + single-process smoke game for a reference
policy file (default reference_claude_opus_v53.py) vs example, timing
BOTH players per step. Single process = the clean ms/step caliber
(RL stage 2 shares the box, noted in reports)."""
import importlib.util
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))
from fastsim import FastCarbon  # noqa: E402

_spec = json.load(open(os.path.join(
    HERE, "env_src", "carbon_challenge-master",
    "zerosum_env", "envs", "carbon", "carbon.json")))
CFG = {}
for _k, _v in _spec["configuration"].items():
    CFG[_k] = _v.get("default") if isinstance(_v, dict) else _v


def load(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    t0 = time.time()
    s.loader.exec_module(m)
    return m, time.time() - t0


def main():
    pol_file = sys.argv[1] if len(sys.argv) > 1 else "reference_claude_opus_v53.py"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    mp, imp_t = load(os.path.join(HERE, pol_file), "bench_pol")
    me, _ = load(os.path.join(HERE, "example.py"), "bench_ex")
    print(f"import {pol_file}: {imp_t:.2f}s")
    random.seed(7919 * seed + 31)
    sim = FastCarbon(CFG).reset(seed=seed)
    agents = [mp.agent, me.agent]
    times = [0.0, 0.0]
    worst = [0.0, 0.0]
    for _ in range(300):
        acts = []
        for p in range(2):
            o = sim.obs(p)
            ts = time.time()
            acts.append(agents[p](o, CFG))
            dt = time.time() - ts
            times[p] += dt
            worst[p] = max(worst[p], dt)
        sim.step(acts[0], acts[1])
    st = sim.get_state()
    print(f"seed={seed} cash: {pol_file}={st['cash'][0]:.2f} example={st['cash'][1]:.2f}")
    print(f"{pol_file}: mean {1000*times[0]/300:.1f} ms/step, worst step {1000*worst[0]:.0f} ms")
    print(f"example: mean {1000*times[1]/300:.2f} ms/step")


if __name__ == "__main__":
    main()
