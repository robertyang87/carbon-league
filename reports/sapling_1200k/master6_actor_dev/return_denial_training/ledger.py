"""One simultaneous transition, evaluated only after the two legal orders have been chosen.

Inputs are full observations and post-step simulator telemetry for labels. Never import this
module to choose an action. Physical-seat arrays index seat 0, then seat 1. The collision
target excludes later deposits, harvesting and absorption, which have their own accounts.
"""

from collections import Counter, defaultdict
from math import isclose

MOVES = {"UP", "RIGHT", "DOWN", "LEFT"}
SPENDING = {"hire_planter", "hire_collector", "plant", "seize"}
DELIVERIES = {"own_delivery", "opponent_worker_delivery_to_this_base"}


def agree(a, b, message):
    if not isclose(a, b, rel_tol=1e-10, abs_tol=1e-6):
        raise ValueError(f"{message}: {a} vs {b}")


def one(before, after, sim, cash_events, seat, own_uid, enemy_uid, own_cmd, enemy_cmd, cfg):
    """Account for the entire board, with focal workers identified separately for diagnostics."""
    if seat not in (0, 1) or after["step"] != before["step"] + 1 or sim.step_count != after["step"]:
        raise ValueError("incorrect seat or nonconsecutive observations")
    if not isinstance(own_cmd, dict) or not isinstance(enemy_cmd, dict):
        raise ValueError("both seat command dictionaries are required")
    orders = (own_cmd, enemy_cmd) if seat == 0 else (enemy_cmd, own_cmd)
    units = [pl[2] for pl in before["players"]]
    survivors = [pl[2] for pl in after["players"]]
    if own_uid not in units[seat] or enemy_uid not in units[1 - seat]:
        raise ValueError("focal workers missing at the start")
    bases = {pos: p for p, pl in enumerate(before["players"]) for pos in pl[1].values()}
    old_trees = {pos: (p, age) for p, pl in enumerate(before["players"]) for pos, age in pl[3].values()}
    expired = {pos for pos, (_, age) in old_trees.items() if age >= cfg["treeLifespan"]}
    live_trees = old_trees.keys() - expired
    locations = {}
    amount = {}
    owners = {}
    kinds = {}
    burn = [0.0, 0.0]
    starting = [0.0, 0.0]
    for p, workers in enumerate(units):
        for uid, (pos, load, kind) in workers.items():
            owners[uid] = p
            kinds[uid] = kind
            starting[p] += load
            cmd = orders[p].get(uid)
            cargo = load * (1 - cfg["moveCost"]) if cmd in MOVES else load
            locations[uid] = sim._move_nbr[cmd][pos] if cmd in MOVES else pos
            amount[uid] = cargo
            burn[p] += load - cargo
    for p, workers in enumerate(survivors):
        for uid, (pos, _, role) in workers.items():
            owners.setdefault(uid, p)
            locations.setdefault(uid, pos)
            amount.setdefault(uid, 0.0)
            kinds.setdefault(uid, role)

    receipts = [Counter(), Counter()]
    direct = {}
    deposits = {}
    trees_from_cargo = [0.0, 0.0]
    for event in cash_events:
        kind, p, val, *detail = event
        receipts[p][kind] += val
        if kind in {"tree_collision_cash", "base_collision_cash"}:
            uid, cell = detail[:2]
            if uid in direct:
                raise ValueError("repeat direct cash payout for one losing worker")
            direct[uid] = (kind, p, cell, val)
        elif kind in DELIVERIES:
            uid, cell = detail
            if uid in deposits:
                raise ValueError("repeat terminal deposit for one worker in one step")
            deposits[uid] = (p, cell, val)
        elif kind == "tree_absorption":
            trees_from_cargo[p] += detail[2]
    for p in (0, 1):
        agree(
            before["players"][p][0] + sum(receipts[p].values()),
            after["players"][p][0],
            "cash flow does not reconcile",
        )

    events = sim.last_collisions
    killed = {e[0] for e in events}
    if len(killed) != len(events):
        raise ValueError("one worker appears twice in the collision journal")
    by_cell = defaultdict(list)
    for uid, pos in locations.items():
        if uid not in killed:
            by_cell[pos].append(uid)
    cross = {name: [0.0, 0.0] for name in ("worker", "tree_cash", "base_cash")}
    friendly = [0.0, 0.0]
    on_board = [0.0, 0.0]
    capped = [0.0, 0.0]
    ground = {}
    given = [0.0, 0.0]
    received_by_worker = [0.0, 0.0]
    losses = {}
    mapped_cash = set()
    incoming = Counter()
    for uid, p, cargo, recipient, winning_seat in events:
        if uid not in locations:
            # A recruited, immediately killed zero-cargo worker has no pre/post entity.
            agree(cargo, 0.0, "unmatched lost cargo")
            continue
        pos = locations[uid]
        agree(cargo, amount[uid], "post-movement cargo of collided worker")
        if p != owners[uid] or uid in survivors[p]:
            raise ValueError("collision identifies the wrong worker")
        given[p] += cargo
        accepted = discarded = 0.0
        ground_before = ground_after = None
        if winning_seat is not None:
            winners = [w for w in by_cell[pos] if owners[w] == winning_seat and kinds[w] == "COLLECTOR"]
            if len(winners) != 1 or recipient != winning_seat:
                raise ValueError("no unique collision recipient")
            received_by_worker[recipient] += cargo
            incoming[winners[0]] += cargo
            destination = "worker"
        elif recipient is None:
            destination = "ground"
            if pos in live_trees or pos in bases:
                raise ValueError("ground drop on occupied landmark")
            level = ground.get(pos, cfg["co2FrmWithered"] if pos in expired else before["carbon"][pos])
            accepted = max(0.0, min(cargo, cfg["maxCellCarbon"] - level))
            discarded = cargo - accepted
            ground[pos] = min(cfg["maxCellCarbon"], level + cargo)
            ground_before, ground_after = level, ground[pos]
            on_board[p] += accepted
            capped[p] += discarded
        else:
            destination = "tree_cash" if pos in live_trees else "base_cash"
            expected = old_trees[pos][0] if pos in live_trees else bases.get(pos)
            if expected != recipient:
                raise ValueError("landmark beneficiary differs")
            if cargo:
                kind, credited, cell, reported = direct[uid]
                if (kind, credited, cell) != (
                    destination.replace("_cash", "_collision_cash"),
                    recipient,
                    pos,
                ):
                    raise ValueError("cash write and collision do not identify the same losing worker")
                agree(reported, cargo, "collision debit differs from cash credit")
                mapped_cash.add(uid)
        if recipient is not None:
            if recipient != p:
                cross[destination][recipient] += cargo
            else:
                friendly[p] += cargo
        losses[uid] = {
            "seat": p,
            "cell": pos,
            "cargo": cargo,
            "destination": destination,
            "beneficiary": recipient,
            "new_ground": accepted,
            "ground_overflow": discarded,
            "ground_before_this_collision_deposit": ground_before,
            "ground_after_this_collision_deposit_before_regrowth_or_tax": ground_after,
        }
    if mapped_cash != set(direct):
        raise ValueError("unmapped positive collision cash event")
    # A collision winner can subsequently deposit or die on an enemy base. Resolve those
    # from actual carried cargo so cap, move fee, smelt and tax are not called 'new harvest'.
    banked = [0.0, 0.0]
    smelt = [0.0, 0.0]
    base_burn = [0.0, 0.0]
    harvest = [0.0, 0.0]
    checked_deposits = set()
    collectors = [sum(w[2] == "COLLECTOR" for w in group.values()) for group in survivors]
    for uid, load in amount.items():
        if uid in killed or kinds[uid] != "COLLECTOR":
            continue
        p, pos = owners[uid], locations[uid]
        load += incoming[uid]
        if pos in bases:
            beneficiary = bases[pos]
            cost = cfg["smeltCost"]
            if beneficiary != p and load <= cost:
                base_burn[p] += load
                if uid in survivors[p]:
                    raise ValueError("enemy-base deletion failed")
            elif load >= cost:
                banked[p] += load
                smelt[p] += cost
                if load > cost:
                    credited, cell, reported = deposits[uid]
                    if (credited, cell) != (beneficiary, pos):
                        raise ValueError("incorrect base receipt")
                    agree(reported, load - cost, "smelted delivery differs")
                    checked_deposits.add(uid)
        elif uid in survivors[p] and orders[p].get(uid) not in MOVES and pos not in live_trees:
            rate = max(cfg["initialCollectRate"] - collectors[p] * cfg["collectDecreaseRate"], 0.0)
            level = cfg["co2FrmWithered"] if pos in expired else before["carbon"][pos]
            harvest[p] += max(0.0, level * rate)
    if checked_deposits != set(deposits):
        raise ValueError("unmapped cash deposit")
    for p in (0, 1):
        ending = sum(w[1] for w in survivors[p].values())
        expected = (
            starting[p]
            - burn[p]
            - given[p]
            + received_by_worker[p]
            - banked[p]
            - base_burn[p]
            + harvest[p]
            - trees_from_cargo[1 - p]
        )
        agree(ending, expected, "carried stock after harvest, collision, settlement and enemy tree tax")

    credited = [sum(v[p] for v in cross.values()) for p in (0, 1)]
    transfer = credited[seat] - credited[1 - seat]
    noncollision = [
        receipts[p]["opponent_worker_delivery_to_this_base"] + trees_from_cargo[p] for p in (0, 1)
    ]
    return {
        "transfer_margin": transfer,
        "collision_cross_received": credited,
        "cross_to_worker": cross["worker"],
        "cross_to_tree_cash": cross["tree_cash"],
        "cross_to_base_cash": cross["base_cash"],
        "friendly_collision_retained": friendly,
        "ground_collision_admitted": on_board,
        "ground_collision_overflow": capped,
        "move_cost_burn": burn,
        "enemy_base_deletion_burn": base_burn,
        "smelt_fee": smelt,
        "real_new_harvest": harvest,
        "gross_cargo_unloaded_by_worker_owner": banked,
        "tree_tax_from_enemy_cargo": trees_from_cargo,
        "tree_from_terrain": [receipts[p]["tree_absorption"] - trees_from_cargo[p] for p in (0, 1)],
        "cash_entries": [dict(x) for x in receipts],
        "actual_spend": [-sum(c[k] for k in SPENDING) for c in receipts],
        "cash_change": [after["players"][p][0] - before["players"][p][0] for p in (0, 1)],
        "cash_income_before_own_spending_and_rounding": [
            sum(v for key, v in c.items() if key not in SPENDING and key != "rounding") for c in receipts
        ],
        "all_phases_cross_flow_margin": transfer + noncollision[seat] - noncollision[1 - seat],
        "own_died": own_uid not in survivors[seat],
        "enemy_died": enemy_uid not in survivors[1 - seat],
        "own_collision": losses.get(own_uid),
        "enemy_collision": losses.get(enemy_uid),
    }
