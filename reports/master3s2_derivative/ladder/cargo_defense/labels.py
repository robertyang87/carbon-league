"""Post-decision collector collision labels; none of the simultaneous orders is a model input."""
from collections import defaultdict
from typing import NamedTuple

ACTIONS = ("NONE", "UP", "RIGHT", "DOWN", "LEFT")
DELTAS = ((0, 0), (-1, 0), (0, 1), (1, 0), (0, -1))
ROLE = {"RECCOLLECTOR": "COLLECTOR", "RECPLANTER": "PLANTER"}
RULES = dict(size=15, recCollectorCost=30, recCollectorIncreaseCost=0, recPlanterCost=30,
             recPlanterIncreaseCost=0, moveCost=0, smeltCost=0, workerLimit=10)


class Arrival(NamedTuple):
    seat: int
    uid: object
    role: str
    cargo: float


def destination(pos, action, size):
    k = ACTIONS.index(action or "NONE")
    r, c = divmod(pos, size)
    dy, dx = DELTAS[k]
    return ((r + dy) % size) * size + (c + dx) % size


def incoming(spot, cargo, command, size, move_cost):
    return destination(spot, command, size), cargo * (1 - move_cost) if command in ACTIONS[1:] else cargo


def arrivals(players, orders, skipped, cfg):
    """Recruitments share per-seat cash and worker quota, with global prices fixed at entry."""
    total = {"COLLECTOR": 0, "PLANTER": 0}
    for _, _, units, _ in players:
        for _, _, role in units.values():
            total[role] += 1
    prices = {role: cfg[prefix+"Cost"] + total[role]*cfg[prefix+"IncreaseCost"]
              for role, prefix in (("COLLECTOR", "recCollector"), ("PLANTER", "recPlanter"))}
    by_cell = defaultdict(list)
    for seat, (cash, bases, units, _) in enumerate(players):
        count = len(units)
        for base, place in bases.items():
            role = ROLE.get(orders[seat].get(base))
            if role is not None and cash >= prices[role] and count < cfg["workerLimit"]:
                count += 1
                cash -= prices[role]
                by_cell[place].append(Arrival(seat, ("recruit", base), role, 0.))
        for uid, (place, cargo, role) in units.items():
            if (seat, uid) == skipped:
                continue
            command = orders[seat].get(uid)
            dest, load = incoming(place, cargo, command, cfg["size"], cfg["moveCost"])
            by_cell[dest].append(Arrival(seat, uid, role, load))
    return by_cell


def collision(group):
    if len(group) == 1:
        return group[0]
    collectors = [a for a in group if a.role == "COLLECTOR"]
    if not collectors:
        return None
    lightest = min(a.cargo for a in collectors)
    winners = [a for a in collectors if a.cargo == lightest]
    return winners[0] if len(winners) == 1 else None


def worker_alive(mover, others, base_owner, smelt_cost):
    group = [mover, *others]
    winner = collision(group)
    # Immediately following collision, an empty or sub-smelt rival worker at a base is deleted.
    combined = sum(a.cargo for a in group)
    alive = winner is mover and not (base_owner == 1-mover.seat and combined <= smelt_cost)
    return alive, winner


def label_actions(players, seat, uid, orders, next_players, configuration=None):
    """Return five conditional labels ordered NONE/UP/RIGHT/DOWN/LEFT, under fixed other orders.

    A hostile-caused death needs both loss with the observed opposing arrivals and survival if
    those arrivals are removed, with all friendly arrivals held fixed. 'loss' is focal cargo
    acquired by the opposing collision winner before base remittance or tree absorption.
    """
    cfg = RULES if configuration is None else configuration
    if seat not in (0, 1) or len(players) != 2 or len(next_players) != 2 or len(orders) != 2:
        raise ValueError("Need both before/after seats and both simultaneous command dictionaries")
    orders = [o or {} for o in orders]
    pos, cargo, role = players[seat][2][uid]
    if role != "COLLECTOR" or cargo < 0:
        raise ValueError("Expected a focal collector and its nonnegative carried carbon")
    incoming_cells = arrivals(players, orders, (seat, uid), cfg)
    owners = {pos: s for s, pl in enumerate(players) for pos in pl[1].values()}
    labels, loss, occupancy, retained, life = [], [], [], [], []
    for action in ACTIONS:
        dest, load = incoming(pos, cargo, action, cfg["size"], cfg["moveCost"])
        other = incoming_cells[dest]
        ally = [a for a in other if a.seat == seat]
        focus = Arrival(seat, uid, role, load)
        alive, winner = worker_alive(focus, other, owners.get(dest), cfg["smeltCost"])
        survives_without_enemy, _ = worker_alive(focus, ally, owners.get(dest), cfg["smeltCost"])
        attributable = survives_without_enemy and not alive
        captured = (load if attributable and winner is not None and winner.role == "COLLECTOR"
                    and winner.seat != seat else 0.)
        labels.append(int(attributable))
        loss.append(float(captured))
        occupancy.append(bool(ally))
        retained.append(float(captured) if owners.get(dest)==seat else 0.)
        life.append(bool(alive))
    actual = ACTIONS.index(orders[seat].get(uid) or "NONE")
    real = uid in next_players[seat][2]
    if life[actual] != real:
        raise ValueError(f"Collector survival differs from official successor: seat={seat}, unit={uid}, action={ACTIONS[actual]}, predicted={life[actual]}, observed={real}")
    return dict(labels=labels, loss=loss, occupancy=occupancy, alive_real=real, actual_index=actual,
                survives=life, cargo_at_own_base_after_enemy_win=retained)


def selftest():
    """Synthetic differential fixtures: derive every true successor with the repository's official Board.next."""
    import json
    from pathlib import Path
    import sys
    repo = next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
    engine = repo/"code/env_src/carbon_challenge-master"
    sys.path.insert(0, str(engine))
    from zerosum_env.envs.carbon.helpers import Board
    from zerosum_env.envs.carbon import idgen
    source = engine/"zerosum_env/envs/carbon/carbon.json"
    cfg = {k:v["default"] for k,v in json.loads(source.read_bytes())["configuration"].items() if isinstance(v,dict) and "default" in v}
    assert all(cfg[k]==v for k,v in RULES.items())

    def worker(cell, mass, kind="COLLECTOR"):
        return [cell, mass, kind]

    def scenario(friends, rivals, their, ours=None, cfg_updates=None, bases=None, cash=30):
        settings = dict(cfg, **(cfg_updates or {}))
        home, away = bases or ({"h0": 18, "h1": 43}, {"e0": 23, "e1": 61})
        players = [[cash, home, friends, {}], [cash, away, rivals, {}]]
        observation = dict(players=players, carbon=[10.]*225, remainingOverageTime=5, player=0, step=45, trees={})
        result = None
        for i, action in enumerate(ACTIONS):
            commands = [dict(ours or {}), dict(their)]
            if action!="NONE":
                commands[0]["focus"]=action
            idgen.reset()
            after = Board(observation,settings,commands).next().observation["players"]
            got = label_actions(players,0,"focus",commands,after,settings)
            if result is None:
                result = got
            else:
                for field in ("labels","loss","occupancy","survives"):
                    assert got[field]==result[field]
            assert got["actual_index"]==i and got["alive_real"]==got["survives"][i]
        return result

    # Focal moves right to the enemy's occupied square. Lightest collector gets the pre-clearing cargo.
    a = scenario({"focus":worker(0,100)},{"other":worker(1,0)}, {})
    assert a["labels"][2]==1 and a["loss"][2]==100 and not a["occupancy"][2]
    b = scenario({"focus":worker(0,100)},{"other":worker(1,100)}, {})
    assert b["labels"][2]==1 and b["loss"][2]==0
    c = scenario({"focus":worker(0,100),"f":worker(1,5)},{"other":worker(2,0)}, {"other":"LEFT"})
    assert c["occupancy"][2] and c["labels"][2]==0 and c["loss"][2]==0  # friendly loss already inevitable
    d = scenario({"focus":worker(0,100)},{"planter":worker(1,0,"PLANTER")}, {})
    assert d["survives"][2] and d["labels"][2]==0
    # One affordable recruitment shared between two bases. Test enemy/own new collector and wrap.
    e = scenario({"focus":worker(0,90)}, {}, {"e0":"RECCOLLECTOR","e1":"RECCOLLECTOR"},
                 bases=({"h":30},{"e0":14,"e1":1}))
    assert e["loss"][4]==90 and e["loss"][2]==0 and e["survives"][2]
    f = scenario({"focus":worker(0,90)}, {}, {}, {"h0":"RECCOLLECTOR","h1":"RECCOLLECTOR"},
                 bases=({"h0":1,"h1":14},{"e":23}))
    assert f["occupancy"][2] and not f["survives"][2] and not f["labels"][2] and not f["occupancy"][4]
    # Unit cap blocks the second recruit even if there is still enough cash.
    rivals={f"p{i}":worker(120+i,0,"PLANTER") for i in range(9)}
    g = scenario({"focus":worker(0,90)}, rivals, {"e0":"RECCOLLECTOR","e1":"RECCOLLECTOR"},
                 bases=({"h":30},{"e0":14,"e1":1}),cash=300)
    assert g["labels"][4] and not g["labels"][2]
    h = scenario({"focus":worker(0,0)}, {}, {}, bases=({"h":30},{"e":1}))
    assert not h["survives"][2] and not h["labels"][2]
    j = scenario({"focus":worker(0,100)},{"other":worker(2,2)}, {"other":"LEFT"},bases=({"h":1},{"e":23}))
    assert j["loss"][2]==100 and j["cargo_at_own_base_after_enemy_win"][2]==100
    k = scenario({"focus":worker(0,100)},{"other":worker(1,95)}, {}, cfg_updates={"moveCost":.1})
    assert k["survives"][2] and not k["labels"][2]


if __name__=="__main__":
    selftest()
    print("50 directed synthetic orders individually checked against official Board.next")
