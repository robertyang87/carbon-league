"""eval_v1: paired-seed evaluation of policy_v1 on FastCarbon.

Methodology (shoulders.md section 6, item 8 -- evaluation discipline):
the local map RNG is unseeded in the original environment, so absolute
scores are NOT comparable across maps. Every seed is therefore played as
a PAIR: same seed (= same map, FastCarbon.reset(seed) is deterministic),
policy_v1 as player 0 in one game and as player 1 in the other. Results
are reported per pair and aggregated over 20 seeds (40 games) vs
reference_juzi, plus 5 pairs (10 games) vs example.py as a sanity check.

Run: python3 eval_v1.py [n_pairs_vs_juzi] [n_pairs_vs_example]
Writes eval_v1_results.jsonl (one line per game) as it goes.
"""

import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

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

_counter = [0]


def load_agent_fn(path, kind):
    """Load an agent function with fresh module state."""
    _counter[0] += 1
    name = f"evalmod_{kind}_{_counter[0]}_{os.getpid()}"
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    if kind == "policy":
        return m.make_agent()
    return m.agent


def play_game(seed, policy_side, opp_kind):
    """One 300-step game. Returns a result dict from policy_v1's view."""
    sim = FastCarbon(CFG).reset(seed=seed)
    policy = load_agent_fn(os.path.join(HERE, "policy_v1.py"), "policy")
    opp = load_agent_fn(os.path.join(HERE, f"{opp_kind}.py"), opp_kind)
    agents = [None, None]
    agents[policy_side] = policy
    agents[1 - policy_side] = opp
    my_time = 0.0
    my_calls = 0
    t0 = time.time()
    for _ in range(300):
        acts = []
        for p in range(2):
            o = sim.obs(p)
            if p == policy_side:
                ts = time.time()
                acts.append(agents[p](o, CFG))
                my_time += time.time() - ts
                my_calls += 1
            else:
                acts.append(agents[p](o, CFG))
        sim.step(acts[0], acts[1])
    wall = time.time() - t0
    st = sim.get_state()
    my_cash = st["cash"][policy_side]
    opp_cash = st["cash"][1 - policy_side]
    stranded = sum(w["carbon"] for w in st["workers"].values()
                   if w["player"] == policy_side)
    return {
        "seed": seed, "opp": opp_kind, "policy_side": policy_side,
        "my_cash": round(my_cash, 2), "opp_cash": round(opp_cash, 2),
        "margin": round(my_cash - opp_cash, 2),
        "win": my_cash > opp_cash, "draw": my_cash == opp_cash,
        "stranded_cargo": round(stranded, 1),
        "my_ms_per_step": round(1000.0 * my_time / max(my_calls, 1), 2),
        "wall_s": round(wall, 1),
    }


def run_pair(args):
    seed, opp_kind = args
    return [play_game(seed, 0, opp_kind), play_game(seed, 1, opp_kind)]


def main():
    n_juzi = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n_ex = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    tasks = [(1000 + i, "reference_juzi") for i in range(n_juzi)]
    tasks += [(2000 + i, "example") for i in range(n_ex)]
    out = open(os.path.join(HERE, "eval_v1_results.jsonl"), "w")
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_pair, t): t for t in tasks}
        for fut in as_completed(futs):
            for res in fut.result():
                out.write(json.dumps(res) + "\n")
                out.flush()
                print(res, flush=True)
    out.close()


if __name__ == "__main__":
    main()
