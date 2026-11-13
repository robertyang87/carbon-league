"""diag_shadow_v2_deaths -- where do the shadow's own-collision deaths
actually happen? Replays shadow vs example (example's unseeded random
is seeded here for a controlled run) and classifies every
own-collision death of the shadow side by cell kind (own base / own
tree / enemy tree / field), step band, group size, and whether the
dead worker's cargo was preserved (group winner = own collector, or
cell pays the cargo back to us via own base/tree).

Usage: python3 diag_shadow_v2_deaths.py SHADOW_FILE SIDE
"""

import importlib.util
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))

from fastsim import FastCarbon  # noqa: E402
from eval_v0_5 import CFG  # noqa: E402
from eval_shadow_v2 import load_agent_fn, _dest  # noqa: E402


def main():
    shadow_file, side = sys.argv[1], int(sys.argv[2])
    random.seed(777)
    sim = FastCarbon(CFG).reset(seed=2000)
    shadow = load_agent_fn(os.path.join(HERE, shadow_file), "policy")
    opp = load_agent_fn(os.path.join(HERE, "example.py"), "opp")
    agents = [None, None]
    agents[side] = shadow
    agents[1 - side] = opp
    own_bases = set()
    stats = Counter()
    cargo_dead = cargo_lost = 0.0
    for step in range(300):
        pre = sim.get_state()
        if not own_bases:
            own_bases = {pos for _bid, pos in pre["bases"][side]}
        acts = []
        for p in range(2):
            acts.append(agents[p](sim.obs(p), CFG))
        sim.step(acts[0], acts[1])
        post = sim.get_state()
        dest = {}
        for wid, w in pre["workers"].items():
            dest[wid] = _dest(w["pos"], (acts[w["player"]] or {}).get(wid))
        for wid, w in post["workers"].items():
            if wid not in pre["workers"]:
                dest[wid] = w["pos"]
        groups = {}
        for wid, d in dest.items():
            groups.setdefault(d, []).append(wid)
        tree_owner = {t["pos"]: t["player"]
                      for t in pre["trees"].values()}
        for wid, w in pre["workers"].items():
            if w["player"] != side or wid in post["workers"]:
                continue
            grp = groups.get(dest[wid], [])
            owners = set()
            for g in grp:
                src = pre["workers"].get(g) or post["workers"].get(g)
                owners.add(src["player"])
            if len(grp) <= 1 or owners != {side}:
                continue
            # own-collision death: classify
            d = dest[wid]
            if d in own_bases:
                kind = "own_base"
            elif tree_owner.get(d) == side:
                kind = "own_tree"
            elif d in tree_owner:
                kind = "enemy_tree"
            else:
                kind = "field"
            band = "early" if step < 100 else (
                "mid" if step < 200 else "late")
            stats[f"{kind}"] += 1
            stats[f"{kind}_{band}"] += 1
            stats["own_collision_deaths"] += 1
            cargo_dead += float(w["carbon"])
            # cargo preserved iff a unique-least own collector wins the
            # group, or the cell pays out to us (own base / own tree)
            coll = []
            for g in grp:
                src = pre["workers"].get(g) or post["workers"].get(g)
                if src["type"] == "COLLECTOR":
                    coll.append((float(src["carbon"]), g))
            preserved = kind in ("own_base", "own_tree")
            if coll:
                mn = min(c for c, _g in coll)
                if sum(1 for c, _g in coll if c == mn) == 1:
                    preserved = True
            if not preserved:
                cargo_lost += float(w["carbon"])
    st = sim.get_state()
    print(json.dumps({
        "shadow_file": shadow_file, "side": side,
        "final_cash": [round(st["cash"][0], 2), round(st["cash"][1], 2)],
        "own_collision_deaths": stats["own_collision_deaths"],
        "by_kind": {k: v for k, v in sorted(stats.items())
                    if k in ("own_base", "own_tree", "enemy_tree", "field")},
        "by_kind_band": {k: v for k, v in sorted(stats.items())
                         if any(k.endswith("_" + b)
                                for b in ("early", "mid", "late"))},
        "cargo_of_own_collision_dead": round(cargo_dead, 1),
        "cargo_est_truly_lost": round(cargo_lost, 1),
        "decorr_stats": getattr(shadow, "stats", None),
    }, indent=1))


if __name__ == "__main__":
    main()
