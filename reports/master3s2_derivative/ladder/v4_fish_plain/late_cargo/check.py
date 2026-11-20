"""Replay one late carrier and change only its commands until its first return."""
import copy
import gzip
import hashlib
import math
from functools import lru_cache
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LADDER = HERE.parents[1]
sys.path.insert(0, str(LADDER / "v4_alphacarbon_decisions"))
from branches import archived, fast, view
from scan import after, dist, feasible, move, new_order, rule

TAPE = LADDER / "v4_observations/increment_0919/replays/bef37c25adeeca8f86eb6a52bcfab97d-1.json.gz"
CARRIER = "player-0-worker-27"
DIRECTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "NONE")


def select(players, orders, avoid_threat):
    pos, load, _ = players[0][2][CARRIER]
    home = players[0][1].values()
    enemy_bases = set(players[1][1].values())
    light = [w[0] for w in players[1][2].values() if w[2] == "COLLECTOR" and w[1] <= load]
    options = []
    for index, command in enumerate(DIRECTIONS):
        dest = move(pos, command)
        if dest in enemy_bases or not feasible(players, orders, CARRIER, command):
            continue
        threatened = any(dist(dest, e) <= 1 for e in light)
        options.append((bool(avoid_threat and threatened), min(dist(dest, b) for b in home), index, command))
    return min(options)[-1] if options else orders.get(CARRIER, "NONE")


def least_tax(players, orders, step, cfg):
    """Compare shortest remaining paths to either friendly base using only public standing trees."""
    position = players[0][2][CARRIER][0]
    bases = set(players[0][1].values())
    enemy_bases = set(players[1][1].values())
    trees = list(players[1][3].values())
    remaining = cfg["episodeSteps"] - 1 - step

    @lru_cache(maxsize=None)
    def tax(cell, elapsed):
        if cell in bases:
            return 0.0
        y, x = divmod(cell, 15)
        count = 0
        for pos, age in trees:
            if age + elapsed - 1 >= cfg["treeLifespan"]:
                continue
            ty, tx = divmod(pos, 15)
            if (x - tx) % 15 in (0, 1, 14) and (y - ty) % 15 in (0, 1, 14):
                count += 1
        return -math.log(max(1e-9, 1 - count * cfg["collectorAbsorptionRate"]))

    @lru_cache(maxsize=None)
    def tail(cell, base, elapsed):
        if cell in bases:
            return 0.0
        length = dist(cell, base)
        if length + elapsed > remaining:
            return math.inf
        successors = [move(cell, a) for a in DIRECTIONS[:4]]
        return min((tax(n, elapsed + 1) + tail(n, base, elapsed + 1) for n in successors
                    if n not in enemy_bases and dist(n, base) < length), default=math.inf)

    options = []
    for index, cmd in enumerate(DIRECTIONS[:4]):
        n = move(position, cmd)
        if n in enemy_bases or not feasible(players, orders, CARRIER, cmd):
            continue
        for base in bases:
            risk = tax(n, 1) + tail(n, base, 1)
            if math.isfinite(risk):
                options.append((round(risk, 9), dist(n, base), index, cmd))
    if options:
        return min(options)[-1]
    return orders.get(CARRIER, "NONE")


def run():
    raw = TAPE.read_bytes()
    tape = json.loads(gzip.decompress(raw))
    frames, cfg = tape["steps"], tape["configuration"]
    assert tape["rewards"] == [7903.0, 8273.56] and tape["statuses"] == ["DONE", "DONE"]
    terrain = frames[0][0]["observation"]["full_carbon"]
    random.seed(cfg["randomSeed"])
    rule.np.random.seed(cfg["randomSeed"])
    env = rule.make("carbon", configuration=cfg)
    env.states[0].observation.full_carbon = copy.deepcopy(terrain)
    sim = rule.econ.FastCarbon(cfg).reset(full_carbon=terrain)
    sim.cash = rule.econ.CashEvents(sim.cash)
    roots = {287, 290, 294, 295}
    results = []
    for t in range(299):
        original = [row["action"] or {} for row in frames[t + 1]]
        assert view(env) == fast(sim) == archived(frames[t])
        if t in roots:
            env.steps[:-1] = [None] * t
            env.logs = []
            saved = rule.official_state.take(env, [0, 0])
            for policy in ("recorded", "shortest", "avoid_light_enemies", "least_tax"):
                rule.official_state.restore(env, saved)
                other = copy.deepcopy(sim)
                trace, bank = [], 0.0
                for i in range(t, 299):
                    players = other.obs(0)["players"]
                    cmds = [{k:v for k,v in (row["action"] or {}).items() if k in players[s][1] or k in players[s][2]}
                            for s,row in enumerate(frames[i+1])]
                    if policy != "recorded" and not bank and CARRIER in players[0][2]:
                        if policy == "least_tax":
                            cmd = least_tax(players, cmds[0], i, cfg)
                        else:
                            cmd = select(players, cmds[0], policy == "avoid_light_enemies")
                        cmds[0] = new_order(cmds[0], CARRIER, cmd)
                    other, _ = after(other, cmds)
                    env.step(cmds, [])
                    assert view(env) == fast(other)
                    if policy == "recorded":
                        assert view(env) == archived(frames[i+1])
                    deposits = [v for kind,s,v,*meta in other.cash.events
                                if kind == "own_delivery" and s == 0 and meta[0] == CARRIER and v > 0]
                    bank += sum(deposits)
                    trace.append(dict(t=i, action=cmds[0].get(CARRIER, "NONE"),
                                      before=players[0][2].get(CARRIER), after=other.obs(0)["players"][0][2].get(CARRIER),
                                      deposited=sum(deposits), cash=list(other.cash)))
                scores = [row.reward for row in env.states]
                assert scores == list(other.cash)
                results.append(dict(t=t, policy=policy, scores=scores, margin=round(scores[0]-scores[1], 2), bank=bank, trace=trace))
            rule.official_state.restore(env, saved)
        sim.cash.events.clear()
        sim.step(*original)
        env.step(original, [])
    assert view(env) == archived(frames[299]) == fast(sim)
    return dict(source_sha256=hashlib.sha256(raw).hexdigest(), history_transitions=299, comparison=results)


if __name__ == "__main__":
    data = run()
    (HERE/"outcomes.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    for r in data["comparison"]:
        print(r["t"], r["policy"], r["margin"], round(r["bank"], 2))
