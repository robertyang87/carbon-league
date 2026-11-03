"""bench_rl -- throughput benchmark for the RL rollout pipeline.

Configs (plain FastCarbon loop, no ledger instrumentation -- the
rollout lower bound; reward bookkeeping overhead is measured again
inside train_rl's real rollouts and reported there):
  A  actor vs actor   both sides NumpyActor(BC weights) on the fast
                      canonical path (self-play cost)
  B  actor vs v0.5    learner fast path + rule policy via obs dicts
                      (pool cost against the strongest sparring partner)

Each config: 2 games single-process, then 8 games over 4 processes.
Reports per-game wall clock and games/hour.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fastsim import FastCarbon  # noqa: E402
import rl_common  # noqa: E402

CFG = rl_common.CFG
BC_WEIGHTS = None


def _bc():
    global BC_WEIGHTS
    if BC_WEIGHTS is None:
        BC_WEIGHTS = rl_common.load_weights_npz(
            os.path.join(HERE, "shadow_weights.npz"))
    return BC_WEIGHTS


def play_actor_vs_actor(seed, side_seed=0):
    sim = FastCarbon(CFG).reset(seed=seed)
    a0 = rl_common.NumpyActor(_bc(), seed=1000 + seed)
    a1 = rl_common.NumpyActor(_bc(), seed=2000 + seed)
    for _ in range(300):
        acts = [None, None]
        for p, act in ((0, a0), (1, a1)):
            acts[p] = act.act_canonical(rl_common.canonical_from_sim(sim, p))
        sim.step(acts[0], acts[1])
    st = sim.get_state()
    return st["cash"]


def play_actor_vs_v05(seed, learner_side):
    sim = FastCarbon(CFG).reset(seed=seed)
    actor = rl_common.NumpyActor(_bc(), seed=3000 + seed)
    v05 = rl_common.load_agent_fn(
        os.path.join(HERE, "policy_v0_5.py"), "policy")
    for _ in range(300):
        acts = [None, None]
        for p in range(2):
            if p == learner_side:
                acts[p] = actor.act_canonical(
                    rl_common.canonical_from_sim(sim, p))
            else:
                acts[p] = v05(sim.obs(p), CFG)
        sim.step(acts[0], acts[1])
    st = sim.get_state()
    return st["cash"]


def _job(args):
    kind, seed = args
    t0 = time.time()
    if kind == "A":
        play_actor_vs_actor(seed)
    else:
        play_actor_vs_v05(seed, seed % 2)
    return time.time() - t0


def bench(kind, label):
    seeds = [5000 + i for i in range(8)]
    t0 = time.time()
    for s in seeds[:2]:
        _job((kind, s))
    single = time.time() - t0
    print(f"{label} 1-proc: 2 games in {single:.1f}s "
          f"-> {single / 2:.2f}s/game, {3600 / (single / 2):.0f} games/h")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        list(ex.map(_job, [(kind, s) for s in seeds]))
    wall = time.time() - t0
    print(f"{label} 4-proc: 8 games in {wall:.1f}s "
          f"-> {8 / wall * 3600:.0f} games/h")
    return 8 / wall * 3600


if __name__ == "__main__":
    bench("A", "actor vs actor (fast path both sides)")
    bench("B", "actor vs v0.5  (fast path + obs-dict policy)")
