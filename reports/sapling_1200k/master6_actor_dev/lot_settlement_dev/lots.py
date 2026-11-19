"""Outcome-side principal accounting for complete, already-recorded games.

Entry t is the observation before decision t. Events use the resulting frame t+1.
A unit of starting cargo keeps its identity through friendly relays and proportional
taxes, and leaves the ledger permanently on its first external transfer or drop.
"""

from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from turning_point_ledger import CashEvents, FastCarbon
from official_game_replay import ENGINE
from zerosum_env.envs.carbon import idgen
from zerosum_env.envs.carbon.helpers import Board

CATEGORIES = ("own_direct", "own_relay", "lost_to_enemy", "tax", "ground_or_cost", "terminal_inventory")
EPS = 1e-11


def check(ok, msg):
    if not ok:
        raise ValueError(msg)


def equal(a, b, msg):
    if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f"{msg}: {a} versus {b}")


def positive_collectors(obs, seat):
    return {
        u: float(v)
        for p, pl in enumerate(obs["players"])
        if seat is None or seat == p
        for u, (_, v, k) in pl[2].items()
        if k == "COLLECTOR" and v > 0
    }


def one_step(sim, orders, seat, verify):
    before = sim.full_observation()
    cfg = sim.config
    workers = {}
    for p, pl in enumerate(before["players"]):
        for u, (pos, v, role) in pl[2].items():
            cmd = orders[p].get(u)
            moving = cmd in sim._move_nbr
            dest = sim._move_nbr[cmd][pos] if moving else pos
            workers[u] = (p, dest, v * (1 - cfg["moveCost"]) if moving else v, role, moving)
    initial = positive_collectors(before, seat)
    counters = sim.worker_counters
    if verify:
        idgen._id_dict.clear()
        for p in (0, 1):
            idgen._id_dict[f"player-{p}-worker-"] = counters[p]
            idgen._id_dict[f"player-{p}-tree-"] = sim.tree_counters[p]
        official = Board(before, cfg, orders).next().observation
    sim.cash = CashEvents(sim.cash)
    sim.step(*orders)
    after = sim.full_observation()
    if verify:
        check(
            all(official[k] == after[k] for k in ("carbon", "trees", "players")),
            "official transition mismatch",
        )
    cash = sim.cash.events
    for k, p, value, *attrs in cash:
        if k.startswith("hire_"):
            uid = f"player-{p}-worker-{counters[p]}"
            counters[p] += 1
            workers[uid] = (p, attrs[1], 0.0, "COLLECTOR" if k == "hire_collector" else "PLANTER", False)
    dead = {u: (p, v, to, win) for u, p, v, to, win in sim.last_collisions}
    survivor = {}
    for u, (p, pos, v, role, moving) in workers.items():
        if u not in dead:
            if pos in survivor:
                raise ValueError("ambiguous postcollision cell")
            survivor[pos] = u
    live_trees = {
        pos: (p, tid)
        for p, pl in enumerate(before["players"])
        for tid, (pos, age) in pl[3].items()
        if age < cfg["treeLifespan"]
    }
    expired = {pos for pl in before["players"] for pos, age in pl[3].values() if age >= cfg["treeLifespan"]}
    bases = {pos: (p, bid) for p, pl in enumerate(before["players"]) for bid, pos in pl[1].items()}
    old_ids = {t for _, t in live_trees.values()}
    active_trees = [
        (p, tid, pos)
        for p, pl in enumerate(after["players"])
        for tid, (pos, _) in pl[3].items()
        if tid in old_ids
    ]
    receipts = {
        e[3]: (e[1], e[2]) for e in cash if e[0] in ("own_delivery", "opponent_worker_delivery_to_this_base")
    }
    collision_cash = {
        e[3]: (e[1], e[2]) for e in cash if e[0] in ("tree_collision_cash", "base_collision_cash")
    }
    post_amount = {u: workers[u][2] for u in survivor.values()}
    ground = {}
    drops = {}
    incoming = Counter()
    for u, p, v, to, win in sim.last_collisions:
        side, pos, postmove, *_ = workers[u]
        equal(postmove, v, "collision before tax")
        if win is not None:
            key = survivor[pos]
            check(
                workers[key][0] == win == to and workers[key][3] == "COLLECTOR", "collision winner identity"
            )
            post_amount[key] += v
            incoming[key] += v
        elif to is None:
            level = ground.get(pos, cfg["co2FrmWithered"] if pos in expired else before["carbon"][pos])
            check(level <= cfg["maxCellCarbon"] + EPS, "unsupported prior ground over cap")
            accepted = min(v, max(0.0, cfg["maxCellCarbon"] - level))
            ground[pos] = level + accepted
            drops[u] = (accepted, v - accepted)
        elif v > 0:
            check(collision_cash[u][0] == to, "unexpected collision cash beneficiary")
            equal(v, collision_cash[u][1], "treasury collision receipt")
    tree_cells = {tid: {pos, *sim._nbr8[pos]} for _, tid, pos in active_trees}
    tax_rates = {}
    tax_actual = Counter()
    collection = Counter()
    pre_tax = {}
    base_actions = {}
    after_stock = {u: v for pl in after["players"] for u, (_, v, k) in pl[2].items() if k == "COLLECTOR"}
    for u in survivor.values():
        p, pos, _, role, moving = workers[u]
        if role != "COLLECTOR":
            continue
        stock = post_amount[u]
        if pos in bases and stock > 0:
            bp, bid = bases[pos]
            fee = cfg["smeltCost"]
            if bp != p and stock <= fee:
                check(u not in after_stock, "incorrect hostile-base removal")
                base_actions[u] = ("enemy_base_deletion", bp, bid, 0.0, stock)
            elif stock >= fee:
                value = stock - fee
                if value > 0:
                    check(receipts[u][0] == bp, "base credited wrong seat")
                    equal(value, receipts[u][1], "base deposit cash")
                base_actions[u] = ("own_bank" if bp == p else "enemy_base_deposit", bp, bid, value, fee)
        if u not in after_stock:
            continue
        trees = [tid for q, tid, pos2 in active_trees if q != p and pos in tree_cells[tid]]
        rate = cfg["collectorAbsorptionRate"]
        check(len(trees) * rate < 1.0, "cannot infer pre-tax stock under complete confiscation")
        closing = after_stock[u]
        gross = closing / (1.0 - len(trees) * rate)
        pre_tax[u] = gross
        tax_rates[u] = [(tree, rate) for tree in trees] if rate > 0 else []
        for tree in trees:
            tax_actual[tree] += gross * rate
        fresh = gross - stock + (sum(base_actions[u][3:]) if u in base_actions else 0.0)
        check(fresh >= -1e-6, "negative inferred harvest")
        if moving:
            equal(fresh, 0.0, "moving worker harvested")
        collection[u] = max(0.0, fresh)
    for e in cash:
        if e[0] == "tree_absorption":
            equal(tax_actual.pop(e[3], 0.0), e[5], "per-tree enemy cargo tax")
    check(all(x <= 1e-7 for x in tax_actual.values()), "unbooked enemy tree income")
    transitions = {}
    for orig, amount in initial.items():
        p, pos, post, _, moved = workers[orig]
        f = post / amount
        actions = []

        def add(kind, reason, holder, receiver, z):
            if z > 0.0:
                actions.append((kind, reason, holder, receiver, z))

        add("ground_or_cost", "move_fee", orig, None, 1 - f)
        holder = orig
        if orig in dead and f > 0:
            _, v, beneficiary, win = dead[orig]
            if win is not None:
                h = survivor[pos]
                if win == p:
                    add("relay", "friendly_worker", orig, h, f)
                    holder = h
                else:
                    add("lost_to_enemy", "enemy_worker", orig, h, f)
                    f = 0.0
            elif beneficiary is None:
                accepted, over = drops[orig]
                add("ground_or_cost", "ground_admitted", orig, None, f * accepted / v)
                add("ground_or_cost", "ground_overflow", orig, None, f * over / v)
                f = 0.0
            else:
                kind, (_, landmark) = ("tree", live_trees[pos]) if pos in live_trees else ("base", bases[pos])
                label = "own" if beneficiary == p else "enemy"
                add(
                    "own_direct" if beneficiary == p else "lost_to_enemy",
                    f"{label}_{kind}_collision",
                    orig,
                    landmark,
                    f,
                )
                f = 0.0
        if f and holder in base_actions:
            reason, beneficiary, bid, value, fees = base_actions[holder]
            total = value + fees
            if value:
                add(
                    "own_direct" if beneficiary == p else "lost_to_enemy",
                    reason,
                    holder,
                    bid,
                    f * value / total,
                )
            add(
                "ground_or_cost",
                "enemy_base_deletion" if reason == "enemy_base_deletion" else "smelt_fee",
                holder,
                bid,
                f * fees / total,
            )
            f = 0.0
        if f:
            check(holder in pre_tax and pre_tax[holder] > 0, "positive principal lost holder")
            rates = tax_rates[holder]
            for tid, rate in rates:
                add("tax", "enemy_tree_tax", holder, tid, f * rate)
            f *= 1 - sum(rate for _, rate in rates)
        equal(sum(e[-1] for e in actions if e[0] != "relay") + f, 1.0, "one-step unit principal")
        transitions[orig] = (actions, holder if f else None, f)
    return initial, transitions, after


class Timeline:
    def __init__(self, values, transitions, final_t):
        self.cargo = values
        self.transitions = transitions
        self.final_t = final_t
        self._summary = {}
        for (t, u), load in values.items():
            if t == final_t:
                self._summary[t, u] = ({"terminal_inventory": 1.0}, 0.0)
        for t in reversed(range(final_t)):
            for u, (actions, successor, remaining) in transitions[t].items():
                grouped = Counter()
                moment = 0.0
                relay = False
                for kind, _, _, _, f in actions:
                    if kind == "relay":
                        relay = True
                        continue
                    if kind == "own_direct" and relay:
                        kind = "own_relay"
                    grouped[kind] += f
                    if kind in ("own_direct", "own_relay"):
                        moment += (t + 1) * f
                if remaining:
                    downstream, later_moment = self._summary[t + 1, successor]
                    for k, f in downstream.items():
                        grouped["own_relay" if relay and k == "own_direct" else k] += f * remaining
                    moment += remaining * later_moment
                equal(sum(grouped.values()), 1.0, "terminal accounting")
                self._summary[t, u] = grouped, moment

    def summary(self, t, uid):
        grouped, moment = self._summary[t, uid]
        f = {kind: grouped.get(kind, 0.0) for kind in CATEGORIES}
        received = f["own_direct"] + f["own_relay"]
        amount = self.cargo[t, uid]
        return dict(
            root_t=t,
            uid=uid,
            root_cargo=amount,
            fractions=f,
            amounts={k: v * amount for k, v in f.items()},
            mean_steps_to_own_cash_conditional=moment / received - t if received else None,
        )

    def events(self, t, uid):
        amount = self.cargo[t, uid]
        remain = 1.0
        relay = False
        holder = uid
        items = []

        def emit(step, kind, reason, worker, recipient, f):
            items.append(
                dict(
                    step=step,
                    kind=kind,
                    reason=reason,
                    holder_uid=worker,
                    recipient_uid=recipient,
                    fraction=f,
                    amount=f * amount,
                    is_terminal_allocation=kind != "relay",
                )
            )

        for i in range(t, self.final_t):
            moves, next_uid, tail = self.transitions[i][holder]
            for k, r, worker, recipient, z in moves:
                if k == "relay":
                    relay = True
                if k == "own_direct" and relay:
                    k = "own_relay"
                emit(i + 1, k, r, worker, recipient, z * remain)
            remain *= tail
            holder = next_uid
            if not remain:
                break
        if remain:
            emit(self.final_t, "terminal_inventory", "held_at_terminal", holder, None, remain)
        equal(sum(v["fraction"] for v in items if v["is_terminal_allocation"]), 1.0, "event mass")
        return items


def from_sim(sim, actions, seat=None, *, verify=True):
    check(sim.step_count == 0 and seat in (None, 0, 1), "expected a game starting at frame zero / valid seat")
    values = {}
    transitions = []
    for t, joint in enumerate(actions):
        v, q, _ = one_step(sim, joint, seat, verify)
        values.update({(t, u): amount for u, amount in v.items()})
        transitions.append(q)
    t = sim.step_count
    values.update({(t, u): amount for u, amount in positive_collectors(sim.full_observation(), seat).items()})
    return Timeline(values, transitions, t)


def build(raw, seat=None, *, verify=True):
    """Use a hash-checked recorded transcript. No policies, searches or future features run."""
    cfg = raw["game_configuration"]
    sim = FastCarbon(cfg).reset(seed=raw["seed"])
    if raw.get("initial_full_carbon") is not None:
        check(raw["initial_full_carbon"] == sim.carbon, "start terrain differs")
    if raw.get("initial_terrain_u8_sha256"):
        check(
            hashlib.sha256(bytes(int(a) for a in sim.carbon)).hexdigest() == raw["initial_terrain_u8_sha256"],
            "start terrain hash differs",
        )
    check(not raw.get("errors") and raw["steps"] == len(raw["actions"]), "invalid or incomplete transcript")
    timeline = from_sim(sim, raw["actions"], seat, verify=verify)
    if "last_post_cash" in raw:
        check(list(sim.cash) == raw["last_post_cash"], "terminal cash mismatch")
    return timeline
