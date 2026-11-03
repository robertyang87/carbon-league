"""measure_belief_mae -- the stage-2 report's "accuracy" (准) metric
for the belief channel: play rl_policy_v2 (rl2_weights.npz, via
rl2_common.NumpyActor2 -- the exact actor the deployed file builds)
against example, and every 10 steps compare the BeliefTracker's
reconstructed carbon map against the simulator's ground-truth map.

Per snapshot:
  coverage     fraction of cells with a belief estimate (>= 0)
  mae_est      MAE over estimated cells (tracker belief vs truth)
  mae_direct   MAE over this step's directly observed cells
               (own collectors' cells + own planters' 4-neighbours;
               sanity anchor -- should be ~0)
  mae_naive    MAE over estimated cells of a memoryless baseline:
               each estimated cell takes the CURRENT observation at
               its first directly-observed symmetric counterpart, or
               the current observed mean if none -- i.e. pure
               four-fold symmetry with no dynamics/memory, what the
               tracker's recursion adds on top of.

Games: seeds 2000 (side 0) and 2001 (side 1) vs example.
Output: belief_mae_results.json. Cheap: 2 games.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src",
                                "carbon_challenge-master"))

import rl2_common  # noqa: E402
import rl_common  # noqa: E402
from belief import _MIRROR, _NBR4  # noqa: E402
from fastsim import FastCarbon  # noqa: E402

N = 225


def direct_mask(canon):
    (_p, _carbon, _cash, _bases, workers, wmask, _trees, _tmask,
     _step, _w, _b) = canon
    p = canon[0]
    mask = np.zeros(N, dtype=bool)
    for j in range(workers.shape[1]):
        if not wmask[p, j]:
            continue
        pos = int(workers[p, j, 0])
        if int(workers[p, j, 2]) == 0:
            mask[pos] = True
        else:
            for c in _NBR4[pos]:
                mask[int(c)] = True
    return mask


def run_game(seed, side, weights):
    cfg = rl_common.CFG
    sim = FastCarbon(cfg).reset(seed=seed)
    me = rl2_common.NumpyActor2(weights, seed=99)
    opp = rl_common.load_agent_fn(os.path.join(HERE, "example.py"),
                                  "opp")
    snaps = []
    for step in range(300):
        canon = rl_common.canonical_from_sim(sim, side)
        acts = [None, None]
        acts[side] = me.act_canonical(canon)[0]
        acts[1 - side] = opp(sim.obs(1 - side), cfg)
        if step % 10 == 0:
            truth = np.asarray(sim.carbon, dtype=np.float64)
            bel = me.tracker.belief.astype(np.float64)
            est = bel >= 0
            dm = direct_mask(canon)
            obs = np.asarray(canon[1], dtype=np.float64)
            obs_mean = float(obs[dm].mean()) if dm.any() else 30.0
            naive = np.full(N, obs_mean)
            for i in range(N):
                for m in _MIRROR[i]:
                    if dm[int(m)]:
                        naive[i] = obs[int(m)]
                        break
            snaps.append({
                "step": step,
                "coverage": round(float(est.mean()), 4),
                "mae_est": round(float(np.abs(bel[est] - truth[est]).mean()), 3)
                if est.any() else None,
                "mae_direct": round(float(np.abs(bel[dm] - truth[dm]).mean()), 3)
                if dm.any() else None,
                "mae_naive": round(float(np.abs(naive[est] - truth[est]).mean()), 3)
                if est.any() else None,
            })
        sim.step(acts[0], acts[1])
    return snaps


def main():
    weights = rl_common.load_weights_npz(
        os.path.join(HERE, "rl2_weights.npz"))
    out = {}
    for seed, side in ((2000, 0), (2001, 1)):
        snaps = run_game(seed, side, weights)
        out[f"seed{seed}_side{side}"] = snaps
        agg = {}
        for k in ("coverage", "mae_est", "mae_direct", "mae_naive"):
            vals = [s[k] for s in snaps if s[k] is not None]
            agg[k] = round(float(np.mean(vals)), 3) if vals else None
        print(f"seed{seed} side{side}: {agg}", flush=True)
    with open(os.path.join(HERE, "belief_mae_results.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
