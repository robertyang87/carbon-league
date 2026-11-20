"""Replay the five shortest routes of worker 15 at t174 against the recorded orders."""
import copy
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import random
import runpy
import sys

HERE = Path(__file__).resolve().parent
DERIVATIVE = HERE.parents[2]
sys.path.insert(0, str(DERIVATIVE.parent / "sapling_1200k/all_top5"))
from economy_extract import CashEvents, FastCarbon

GAME = "d43d676041c83c5d9380f9daf394d2ce"
UID = "player-0-worker-15"
OTHER = "player-1-worker-10"
START = 174
END = 180


def visible_state(frame):
    return [frame[0]["observation"][field] for field in ("step", "players", "trees", "full_carbon")]


def run():
    referee = runpy.run_path(str(DERIVATIVE / "freight_response/run.py"))
    snap = referee["official_state"]
    raw = (HERE.parents[1] / "replays" / (GAME + "-1.json.gz")).read_bytes()
    data = json.loads(gzip.decompress(raw))
    frames, cfg = data["steps"], data["configuration"]
    random.seed(cfg["randomSeed"])
    referee["np"].random.seed(cfg["randomSeed"])
    env = referee["make"]("carbon", configuration=cfg)
    terrain = frames[0][0]["observation"]["full_carbon"]
    env.states[0].observation.full_carbon = copy.deepcopy(terrain)
    start = FastCarbon(cfg).reset(full_carbon=terrain)
    for t in range(START):
        actions = [seat["action"] or {} for seat in frames[t+1]]
        env.step(actions, [])
        start.step(*actions)
        assert visible_state(env.states) == visible_state(frames[t+1])
        assert start.obs(0)["players"] == env.states[0].observation.players
    env.steps[:-1] = [None] * START
    env.logs = []
    initial = snap.take(env, [0, 0])
    roster = frames[START][0]["observation"]["players"][0]
    assert cfg["size"] == 15 and set(roster[1].values()) == {64, 160}
    assert roster[2][UID][0] == 149
    # On this 15x15 torus the unique nearest base is cell 160 (down 1, left 4).
    routes = sorted(set(itertools.permutations(["DOWN"] + ["LEFT"] * 4)))
    outcomes = []
    for route in [None] + routes:
        snap.restore(env, initial)
        sim = copy.deepcopy(start)
        sim.cash = CashEvents(sim.cash)
        trace = []
        for i, t in enumerate(range(START, END)):
            actions = [dict(seat["action"] or {}) for seat in frames[t+1]]
            if route is not None and UID in sim.workers:
                if i < len(route):
                    actions[0][UID] = route[i]
                else:
                    actions[0].pop(UID, None)
            sim.cash.events.clear()
            env.step(actions, [])
            sim.step(*actions)
            obs = env.states[0].observation
            assert obs.players == sim.obs(0)["players"] and obs.trees == sim.obs(0)["trees"]
            assert list(sim.carbon) == obs.full_carbon
            if route is None:
                assert visible_state(env.states) == visible_state(frames[t+1])
            receipts = [e for e in sim.cash.events if e[0] == "own_delivery" and e[2] > 0]
            trace.append(dict(step=t+1, our=obs.players[0][2].get(UID), their=obs.players[1][2].get(OTHER),
                              cash=list(sim.cash), deliveries=receipts))
        outcomes.append(dict(route=route, trace=trace, final_cash_difference=round(sim.cash[0]-sim.cash[1],2)))
    return dict(match=GAME, raw_sha256=hashlib.sha256(raw).hexdigest(), start=START,end=END,
                held_fixed="all recorded commands except worker 15; waits after arrival", outcomes=outcomes)


if __name__ == "__main__":
    report = run()
    (HERE / "cargo_return.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    for outcome in report["outcomes"]:
        print(outcome["route"], outcome["final_cash_difference"],
              [(e[3], round(e[2],2)) for row in outcome["trace"] for e in row["deliveries"] if e[3] in (UID, OTHER)])
