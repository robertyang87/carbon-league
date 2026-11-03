"""diag_v04_calib: calibrate the tree-warfare value gates against
GROUND TRUTH. Plays the 8 paired juzi games with the instrumented
policy_v0_4 (plant_home_on=0 => v0.3 behaviour) and, for every tree
juzi seizes from us, pairs
  * the policy's registration-gate estimate at flip time
    (probe flip_vals, chronological), with
  * the tree's ACTUAL subsequent absorption while juzi owned it
    (summed from trees_dict[tid][1] step by step in this harness).
Question: is 150 a miscalibrated gate, a biased estimator, or were
the trees really not worth recovering?

  python3 diag_v04_calib.py [policy_file]
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from statistics import mean, median

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eval_v0_4 as E  # CFG, load_agent_fn
from fastsim import FastCarbon


def play(task):
    seed, side, policy_file = task
    sim = FastCarbon(E.CFG).reset(seed=seed)
    policy = E.load_agent_fn(os.path.join(HERE, policy_file), "policy")
    opp = E.load_agent_fn(os.path.join(HERE, "reference_juzi.py"), "juzi")
    agents = [None, None]
    agents[side] = policy
    agents[1 - side] = opp
    flips = []          # dicts: step, tid, actual, held, back
    open_flips = {}     # tid -> flip dict (still juzi-owned)
    for step in range(300):
        pre = sim.get_state()
        acts = [agents[p](sim.obs(p), E.CFG) for p in range(2)]
        sim.step(acts[0], acts[1])
        post = sim.get_state()
        # absorption this step for currently-tracked flipped trees
        for tid, f in open_flips.items():
            t = post["trees"].get(tid)
            if t is not None and t["player"] == 1 - side:
                f["actual"] += float(
                    post["trees_dict"].get(tid, [None, 0])[1])
                f["held"] += 1
        # flips resolved this step (back to us or died)
        for tid in list(open_flips):
            t = post["trees"].get(tid)
            if t is None or t["player"] == side:
                open_flips[tid]["back"] = t is not None
                flips.append(open_flips.pop(tid))
        # new flips: mine in pre, juzi's in post, same tid
        for tid, t in post["trees"].items():
            if t["player"] == 1 - side and tid in pre["trees"] \
                    and pre["trees"][tid]["player"] == side:
                open_flips[tid] = {"step": step + 1, "tid": tid,
                                   "actual": 0.0, "held": 0, "back": False}
    flips.extend(open_flips.values())
    pol = getattr(policy, "policy", None)
    tw = dict(pol.tw) if pol is not None else {}
    return {"seed": seed, "side": side,
            "cash": [round(post["cash"][0], 2), round(post["cash"][1], 2)],
            "flip_vals": tw.get("flip_vals", []),
            "tw": {k: v for k, v in tw.items() if k != "flip_vals"},
            "flips": flips}


def main():
    policy_file = sys.argv[1] if len(sys.argv) > 1 else "policy_v0_4.py"
    tasks = [(s, side, policy_file)
             for s in (1000, 1001, 1002, 1003) for side in (0, 1)]
    games = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(play, t): t for t in tasks}
        for fut in as_completed(futs):
            g = fut.result()
            games.append(g)
            print(f"seed{g['seed']} P{g['side']}: cash {g['cash']} "
                  f"flips {len(g['flips'])} est {len(g['flip_vals'])}",
                  flush=True)
    est_all, act_all = [], []
    print("\nper-flip (estimate -> actual post-flip income for juzi):")
    for g in sorted(games, key=lambda x: (x["seed"], x["side"])):
        fl = sorted(g["flips"], key=lambda f: f["step"])
        ev = g["flip_vals"]
        for i, f in enumerate(fl):
            e = ev[i] if i < len(ev) else None
            est_all.append(e)
            act_all.append(f["actual"])
            print(f"  seed{g['seed']} P{g['side']} step{f['step']:3d} "
                  f"est {e if e is not None else float('nan'):6.1f} -> "
                  f"actual {f['actual']:7.1f}  held {f['held']:3d} steps"
                  f"{'  (recovered)' if f['back'] else ''}")
    pairs = [(e, a) for e, a in zip(est_all, act_all) if e is not None]
    if pairs:
        es = [p[0] for p in pairs]
        ac = [p[1] for p in pairs]
        print(f"\npaired flips: {len(pairs)}")
        print(f"estimate: mean {mean(es):.1f} median {median(es):.1f} "
              f"max {max(es):.1f}")
        print(f"actual  : mean {mean(ac):.1f} median {median(ac):.1f} "
              f"max {max(ac):.1f}")
        for thr in (150, 100, 75, 50, 40, 30):
            print(f"  actual >= {thr:3d}: "
                  f"{sum(1 for a in ac if a >= thr)}/{len(ac)}   "
                  f"swing(2x) >= {thr}: "
                  f"{sum(1 for a in ac if 2 * a >= thr)}/{len(ac)}")
        print(f"mean actual/estimate ratio: "
              f"{mean([a / max(e, 1.0) for e, a in pairs]):.2f}")


if __name__ == "__main__":
    main()
