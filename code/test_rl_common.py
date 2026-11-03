"""test_rl_common -- prove canonical_from_sim is bit-identical to
shadow_alpha._canonical(sim.obs(p)) on live game states (both players,
every step of two full games), so PPO rollouts can use the fast path
with zero train/eval feature drift."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from fastsim import FastCarbon  # noqa: E402
import rl_common  # noqa: E402
import shadow_alpha  # noqa: E402

CFG = rl_common.CFG


def check_game(seed, agents):
    sim = FastCarbon(CFG).reset(seed=seed)
    for step in range(300):
        for p in range(2):
            fast = rl_common.canonical_from_sim(sim, p)
            ref = shadow_alpha._canonical(sim.obs(p))
            names = ("player", "carbon", "cash", "bases", "workers",
                     "wmask", "trees", "tmask", "step")
            for i, nm in enumerate(names):
                a, b = fast[i], ref[i]
                if isinstance(a, np.ndarray):
                    assert np.array_equal(a, b), \
                        f"seed {seed} step {step} p{p} field {nm} differs"
                else:
                    assert a == b, \
                        f"seed {seed} step {step} p{p} field {nm}: {a} vs {b}"
            assert fast[9] == ref[9], f"wids differ {fast[9]} vs {ref[9]}"
            assert fast[10] == ref[10], f"bids differ {fast[10]} vs {ref[10]}"
        acts = [agents[p](sim.obs(p), CFG) for p in range(2)]
        sim.step(acts[0], acts[1])
    print(f"seed {seed}: 300 steps x 2 players bit-identical")


def main():
    shadow = shadow_alpha.make_agent()
    opp = rl_common.load_agent_fn(os.path.join(HERE, "example.py"), "opp")
    check_game(777, [shadow, opp])
    shadow2 = shadow_alpha.make_agent()
    v05 = rl_common.load_agent_fn(
        os.path.join(HERE, "policy_v0_5.py"), "policy")
    check_game(1234, [v05, shadow2])
    print("ALL EQUAL")


if __name__ == "__main__":
    main()
