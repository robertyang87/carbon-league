"""eval_s4: structure line S4 -- camp verdict.

Three-way paired comparison (same seed = same map, both sides):
  s4nc   fork_s4_nocamp  vs base_v53   seeds 1000-1003 x2
  s4cf   fork_s4_campfix vs base_v53   seeds 1000-1003 x2
  s4hh   winner          vs runner-up  seeds 1000-1001 x2   (phase 2)

Methodology = eval_v53 (FastCarbon, same RNG protocol
random.seed(7919*seed + 104729*policy_side + 31), same per-player
ledger from state diffs). Additions: camp instrumentation read from
the policy module's _BOT after every act() call --
  camper_steps     sum over steps of len(bot.camp)
  collector_steps  sum over steps of own collectors alive (state)
  camp_level       bot.camp_level at steps 50/100/200/300 + first step
                   it drops below PARAMS['campers'] (downgrade_step)
  camp_lost_final  bot.camp_lost at game end; camp_deaths = len(bot.camp_deaths)
  opp_deliv_seen   bot.en_deliv at game end (deliveries BY the opponent)
  intercept_events / intercept_cargo: from state diffs -- an enemy
                   collector dies on a cell where one of my posted
                   campers stands in the post state and that camper's
                   cargo rose; cargo counted = min(camper gain, dead
                   enemy's pre-step cargo).
  camp_post_gain   all positive cargo deltas of posted campers that
                   stayed on the same cell (interception + collecting
                   on the post cell, upper bound of camp-side gains)

Appends one JSON line per game to eval_s4_results.jsonl.
Usage: python3 eval_s4.py [s4nc|s4cf|s4hh ...]   (default: s4nc s4cf)
For s4hh the pairing is read from s4_hh_pairing.txt ("winner runnerup").
"""

import importlib.util
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))

from fastsim import FastCarbon  # noqa: E402
import eval_v53  # noqa: E402  (CFG, rng_seed, ledger helpers)

CFG = eval_v53.CFG
BASE = "base_v53.py"
NOCAMP = "fork_s4_nocamp.py"
CAMPFIX = "fork_s4_campfix.py"

JOBS = []
for _seed in (1000, 1001, 1002, 1003):
    for _side in (0, 1):
        JOBS.append(("s4nc", NOCAMP, BASE, _seed, _side))
        JOBS.append(("s4cf", CAMPFIX, BASE, _seed, _side))

JOBS_HH = []
_pairing_path = os.path.join(HERE, "s4_hh_pairing.txt")
if os.path.exists(_pairing_path):
    _w, _r = open(_pairing_path).read().split()
    for _seed in (1000, 1001):
        for _side in (0, 1):
            JOBS_HH.append(("s4hh", _w, _r, _seed, _side))

JOBS_ALL = JOBS + JOBS_HH

_counter = [0]


def load_agent(path):
    """Return (agent_fn, module). v53-family files expose module-level
    agent(); camp state is read off module._BOT after each call."""
    _counter[0] += 1
    name = f"evals4_{_counter[0]}_{os.getpid()}"
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m.agent, m


def new_camp():
    return dict(camper_steps=0, collector_steps=0, camp_level_marks={},
                downgrade_step=None, camp_lost_final=0, camp_deaths=0,
                opp_deliv_seen=0.0, intercept_events=0,
                intercept_cargo=0.0, camp_post_gain=0.0, has_camp=False)


def play_game(task):
    tag, a_file, b_file, seed, a_side = task
    random.seed(eval_v53.rng_seed(seed, a_side))
    sim = FastCarbon(CFG).reset(seed=seed)
    fa, ma = load_agent(os.path.join(HERE, a_file))
    fb, mb = load_agent(os.path.join(HERE, b_file))
    agents = [None, None]
    mods = [None, None]
    agents[a_side] = fa
    agents[1 - a_side] = fb
    mods[a_side] = ma
    mods[1 - a_side] = mb
    led = [eval_v53.new_ledger(), eval_v53.new_ledger()]
    camp = [new_camp(), new_camp()]
    times = [0.0, 0.0]
    worst_a = 0.0
    comp = {}
    t0 = time.time()
    for step in range(1, 301):
        pre = sim.get_state()
        acts = []
        camp_now = [{}, {}]  # player -> {wid: post_cell} snapshot after act
        for p in range(2):
            o = sim.obs(p)
            ts = time.time()
            acts.append(agents[p](o, CFG))
            dt = time.time() - ts
            times[p] += dt
            if p == a_side:
                worst_a = max(worst_a, dt)
            bot = getattr(mods[p], "_BOT", None)
            if bot is not None and hasattr(bot, "camp"):
                C = camp[p]
                C["has_camp"] = True
                C["camper_steps"] += len(bot.camp)
                camp_now[p] = dict(bot.camp)
                lvl = bot.camp_level
                if step in (50, 100, 200, 300):
                    C["camp_level_marks"][str(step)] = lvl
                full = bot.P["campers"]
                if (C["downgrade_step"] is None and lvl is not None
                        and full > 0 and lvl < full):
                    C["downgrade_step"] = step
        for p in range(2):
            camp[p]["collector_steps"] += sum(
                1 for w in pre["workers"].values()
                if w["player"] == p and w["type"] == "COLLECTOR")
        sim.step(acts[0], acts[1])
        post = sim.get_state()
        if step in (100, 200, 300):
            comp[step] = [eval_v53._comp(post, 0), eval_v53._comp(post, 1)]
        # camp interception / post gains from state diffs
        for p in range(2):
            C = camp[p]
            if not camp_now[p]:
                continue
            dead_foe = {}
            for wid, w in pre["workers"].items():
                if w["player"] != p and wid not in post["workers"]:
                    dead_foe.setdefault(w["pos"], []).append(w)
            for wid, post_cell in camp_now[p].items():
                wpre = pre["workers"].get(wid)
                wpost = post["workers"].get(wid)
                if wpre is None or wpost is None:
                    continue
                gain = wpost["carbon"] - wpre["carbon"]
                if gain <= 0:
                    continue
                if wpost["pos"] == wpre["pos"]:
                    C["camp_post_gain"] += gain
                for foe in dead_foe.get(wpost["pos"], []):
                    if foe["type"] == "COLLECTOR" and foe["carbon"] > 0:
                        C["intercept_events"] += 1
                        C["intercept_cargo"] += min(gain, foe["carbon"])
        for p in range(2):
            L = led[p]
            price = eval_v53._plant_price(pre["trees"], p)
            rs = ps = ss = 0.0
            for tid, t in post["trees"].items():
                if t["player"] != p:
                    continue
                L["tree_income"] += float(
                    post["trees_dict"].get(tid, [None, 0])[1])
                if tid not in pre["trees"]:
                    L["plant_count"] += 1
                    ps += price
                elif pre["trees"][tid]["player"] != p:
                    L["seize_count"] += 1
                    ss += float(CFG["seizeCost"])
            for wid, w in post["workers"].items():
                if w["player"] == p and wid not in pre["workers"]:
                    L["recruit_count"] += 1
                    rs += float(CFG["recCollectorCost"])
            for wid, w in pre["workers"].items():
                if w["player"] == p and wid not in post["workers"]:
                    L["deaths"] += 1
                    L["death_cargo"] += float(w["carbon"])
            L["recruit_spend"] += rs
            L["plant_spend"] += ps
            L["seize_spend"] += ss
            L["deposits"] += (post["cash"][p] - pre["cash"][p]) + rs + ps + ss \
                - float(sum(post["trees_dict"].get(tid, [None, 0])[1]
                            for tid, t in post["trees"].items()
                            if t["player"] == p))
    wall = time.time() - t0
    st = sim.get_state()
    for p in range(2):
        bot = getattr(mods[p], "_BOT", None)
        if bot is not None and hasattr(bot, "camp"):
            camp[p]["camp_lost_final"] = bot.camp_lost
            camp[p]["camp_deaths"] = len(bot.camp_deaths)
            camp[p]["opp_deliv_seen"] = round(bot.en_deliv, 1)
        for k in ("intercept_cargo", "camp_post_gain"):
            camp[p][k] = round(camp[p][k], 1)
        led[p]["stranded"] = round(sum(
            w["carbon"] for w in st["workers"].values()
            if w["player"] == p), 1)
        for k in ("deposits", "tree_income", "recruit_spend", "plant_spend",
                  "seize_spend", "death_cargo"):
            led[p][k] = round(led[p][k], 1)
    return {
        "tag": tag, "seed": seed, "policy_side": a_side,
        "a_file": a_file, "b_file": b_file,
        "my_cash": round(st["cash"][a_side], 2),
        "opp_cash": round(st["cash"][1 - a_side], 2),
        "margin": round(st["cash"][a_side] - st["cash"][1 - a_side], 2),
        "win": st["cash"][a_side] > st["cash"][1 - a_side],
        "a_ms_per_step": round(1000.0 * times[a_side] / 300, 2),
        "b_ms_per_step": round(1000.0 * times[1 - a_side] / 300, 2),
        "a_worst_step_ms": round(1000.0 * worst_a, 0),
        "wall_s": round(wall, 1),
        "ledger_pol": led[a_side], "ledger_opp": led[1 - a_side],
        "camp_pol": camp[a_side], "camp_opp": camp[1 - a_side],
        "comp": {str(k): v for k, v in comp.items()},
    }


def main():
    tags = sys.argv[1:] or ["s4nc", "s4cf"]
    jobs = [j for j in JOBS_ALL if j[0] in tags]
    out_path = os.path.join(HERE, "eval_s4_results.jsonl")
    out = open(out_path, "a")
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(play_game, t): t for t in jobs}
        for fut in as_completed(futs):
            res = fut.result()
            out.write(json.dumps(res) + "\n")
            out.flush()
            print(res["tag"], res["seed"], res["policy_side"],
                  res["my_cash"], res["opp_cash"], res["margin"],
                  f"a_ms={res['a_ms_per_step']} b_ms={res['b_ms_per_step']}",
                  flush=True)
    out.close()


if __name__ == "__main__":
    main()
