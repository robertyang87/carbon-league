"""Paired one-order interventions on sealed Sapling-v1 games, with both teams responding.

Run the two arms in separate single-thread, CPU-pinned processes. Official scores come
from complete original-engine trajectories. Ten-turn and terminal balances below track
the specific cargo on the selected worker, alongside all subsequent harvest and deposits.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import importlib.util
import json
from math import isclose
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = BASE.parents[1]
RECOVERY = BASE / "root_recovery_branches"
HORIZON = HERE.parent / "freight_horizon"
sys.path.insert(0, str(RECOVERY))
import live
from branches import begin, clean, public, utility

sys.path.insert(0, str(HORIZON))
from data import FastCarbon, FreightCash, add_step, check_cfg
from turning_point_ledger import CashEvents

V1 = "9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0"
ACTIONS = {"NONE", "UP", "DOWN", "LEFT", "RIGHT"}


def read(path):
    b = path.read_bytes()
    return json.loads(gzip.decompress(b) if path.suffix == ".gz" else b)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retain(path, obj):
    if path.is_file():
        require(read(path) == obj, f"immutable receipt changed: {path}")
    else:
        live.save(path, obj)


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


@contextmanager
def lock(path):
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def register(planpath, expected, index):
    require(sha(planpath) == expected, "plan digest differs from frozen CLI digest")
    plan = read(planpath)
    live.PACK = ROOT / plan["parent_path"]
    live.PACK_HASH = V1
    require(sha(live.PACK) == plan["parent_sha256"] == V1, "incorrect v1 pack")
    require(sha(live.SOURCE) == plan["examiner_sha256"] == live.MASTER_SHA, "incorrect native Master3s2")
    native = subprocess.check_output(["git", "-C", str(ROOT), "show", "HEAD:sources/master3s2/master3s2.py"])
    require(
        hashlib.sha256(native).hexdigest() == live.MASTER_SHA, "examiner differs from repository original"
    )
    if "selection_path" in plan:
        require(sha(ROOT / plan["selection_path"]) == plan["selection_sha256"], "selection changed")
    splits = {}
    roots = set()
    for r in plan["roots"]:
        key = (r["seed"], r["seat"], r["t"], r["uid"], r["rng_seed"])
        require(key not in roots and r["t"] > 0 and r["seat"] in (0, 1), "duplicate or invalid root")
        require(splits.setdefault(r["seed"], r["split"]) == r["split"], "split overlaps by map")
        roots.add(key)
        require(r["old"] != r["alternative"] and {r["old"], r["alternative"]} <= ACTIONS, "invalid action")
    require(0 <= index < len(plan["roots"]), "case index out of range")
    r = plan["roots"][index]
    src = ROOT / r["source"]
    require(sha(src) == r["source_sha256"], "original trajectory digest mismatch")
    raw = read(src)
    seat, t = r["seat"], r["t"]
    require(
        (raw["seed"], raw["candidate_seat"], raw["pack_sha256"]) == (r["seed"], seat, V1), "source identity"
    )
    require(
        t < len(raw["actions"]) and clean(raw["actions"][t][seat]).get(r["uid"], "NONE") == r["old"],
        "archived original action mismatch",
    )
    official_path = src.parent / "official_replay.json"
    games = read(official_path)["runs"][src.parent.name]
    require(games["pack_sha256"] == V1, "source official pack")
    (game,) = [g for g in games["games"] if (g["seed"], g["seat"]) == (r["seed"], seat)]
    raw["initial_terrain_u8_sha256"] = game["map_sha256"]
    case = dict(
        seed=r["seed"],
        seat=seat,
        root_t=t,
        family="safe_own_courier_return",
        source_dir=str(Path(r["source"]).parent),
        record=dict(seed=r["seed"], seat=seat, source=r["source"], source_sha256=r["source_sha256"]),
        root=dict(t=t, worker=r["uid"], old=r["old"], alternative=r["alternative"], tree=""),
        split=r["split"],
        plan_sha256=expected,
        rng_seed=r["rng_seed"],
        map_sha256=game["map_sha256"],
        source_official_sha256=sha(official_path),
        driver_sha256=sha(Path(__file__)),
        native_responder_sha256=sha(Path(live.__file__)),
    )
    wd = HERE / "branches" / f"case_{index:02d}_{r['seed']}_{seat}_{t}"
    return r, case, raw, wd


def initialize(wd, r, case, raw):
    wd.mkdir(parents=True, exist_ok=True)
    with lock(wd / ".prepare.lock"):
        sim = FastCarbon(raw["game_configuration"]).reset(seed=r["seed"])
        for joint in raw["actions"][: r["t"]]:
            sim.step(*joint)
        seat, uid = r["seat"], r["uid"]
        worker = sim.workers.get(uid)
        require(
            worker and worker.is_collector and worker.player == seat and worker.carbon > 0,
            "no loaded own courier",
        )
        orders = clean(raw["actions"][r["t"]][seat])

        def end(u, cmd):
            pos = sim.workers[u].pos
            return pos if cmd == "NONE" else sim._move_nbr[cmd][pos]

        dest = end(uid, r["alternative"])
        other = {end(u, orders.get(u, "NONE")) for u in sim.player_workers[seat] if u != uid}
        hires = {pos for bid, pos in sim.bases[seat] if orders.get(bid) in {"RECPLANTER", "RECCOLLECTOR"}}
        require(
            dest not in other | hires and dest not in {p for _, p in sim.bases[1 - seat]},
            "illegal or conflicting alternative",
        )
        own_home = [p for _, p in sim.bases[seat]]

        def distance(pos):
            n = sim.size

            def axis(i, j):
                d = abs(i - j)
                return min(d, n - d)

            return min(axis(pos % n, p % n) + axis(pos // n, p // n) for p in own_home)

        before, old, new = map(distance, [worker.pos, end(uid, r["old"]), dest])
        require(old == new == before - 1, "both registered routes must reduce own home distance equally")
        retain(
            wd / "root_input.json.gz",
            dict(
                t=r["t"],
                seed=r["seed"],
                seat=seat,
                uid=uid,
                cargo=worker.carbon,
                current=worker.pos,
                v1_landing=end(uid, r["old"]),
                alternative_landing=dest,
                home_distance=before,
                own_public=sim.obs(seat),
                joint_at_root=raw["actions"][r["t"]],
            ),
        )
        retain(wd / "live_protocol.json", case)
        rng = np.random.RandomState(r["rng_seed"]).get_state()
        shared = dict(
            mt_algorithm=rng[0],
            mt_keys=rng[1].tolist(),
            mt_index=rng[2],
            has_normal=rng[3],
            stored_normal=rng[4],
            source_sha256=live.MASTER_SHA,
        )
        dest = wd / "live_shared_init.json"
        if not dest.exists():
            master = live.module(live.SOURCE, "courier_master_initialization")
            require(
                master._TEMP == 1 and master._SEED is None and master._SEARCH_OK and master._RX,
                "examiner search or original temperature not active",
            )
            retain(dest, {**shared, "warm_cost": list(master._S_COST)})
        snap = read(dest)
        require(
            all(snap.get(k) == v for k, v in shared.items()) and len(snap["warm_cost"]) > 0,
            "paired stream differs",
        )


def equal(x, y, context):
    require(isclose(x, y, abs_tol=1e-5, rel_tol=1e-8), f"{context}: {x} vs {y}")


def summarize_arm(wd, r, raw, arm):
    record = read(wd / f"live_{arm}.json")
    action_path = wd / record["action_tape"]
    tail = read(action_path)
    require(
        record["action_sha256"] == sha(action_path)
        and record["search_calls"] == len(tail) - 1
        and record["statuses"] == ["DONE", "DONE"]
        and record["protocol_sha256"] == sha(wd / "live_protocol.json")
        and record["shared_init_sha256"] == sha(wd / "live_shared_init.json")
        and record["source_sha256"] == live.MASTER_SHA
        and record["pack_sha256"] == V1,
        "unsealed or partial branch",
    )
    seat, uid, t = r["seat"], r["uid"], r["t"]
    env = begin(raw)
    full = raw["actions"][:t] + tail
    cfg = raw["game_configuration"]
    check_cfg(cfg)
    sim = FastCarbon(cfg).reset(seed=r["seed"])
    book = FastCarbon(cfg).reset(seed=r["seed"])
    v1 = live.module(live.PACK, "courier_v1_full_command_check")
    for i, joint in enumerate(full[:t]):
        require(clean(v1.agent(public(env, seat), cfg)) == clean(joint[seat]), f"wrong v1 prefix at {i}")
        env.step(joint)
        sim.step(*joint)
        book.step(*joint)
    at_root = sim.workers[uid].carbon
    sim.cash = FreightCash(sim.cash)
    book.cash = CashEvents(book.cash)
    tagged = at_root
    holder = uid
    settled = Counter()
    cumulative = Counter()
    team = [Counter(), Counter()]
    trace = []
    checkpoints = {}
    for i, joint in enumerate(tail):
        step = t + i
        require(not env.done, "extra order after official terminal")
        v1_orders = clean(v1.agent(public(env, seat), cfg))
        if i == 0:
            require(clean(joint[1 - seat]) == clean(raw["actions"][t][1 - seat]), "root opponent changed")
            if arm == "alternative":
                if r["alternative"] == "NONE":
                    v1_orders.pop(uid, None)
                else:
                    v1_orders[uid] = r["alternative"]
        require(v1_orders == clean(joint[seat]), f"nonfocal order or subsequent v1 mismatch at {step}")
        pre = {u: (w.pos, w.carbon, w.player) for u, w in sim.workers.items()}
        env.step(joint)
        event = add_step(sim, joint, env.states[0].observation.players)
        book.cash.events.clear()
        book.step(*joint)
        require(list(book.cash) == list(sim.cash), "instruments disagree on both seat balances")
        for name, side, value, *_ in book.cash.events:
            team[side][name] += value
        lost = {e[0]: e[1:] for e in sim.last_collisions}
        winners = {w.pos: u for u, w in sim.workers.items() if w.is_collector}
        gains = Counter()
        for dead, (p, value, beneficiary, winner) in lost.items():
            if value and winner is not None:
                pos = pre[dead][0]
                cmd = joint[p].get(dead)
                landed = sim._move_nbr[cmd][pos] if cmd in sim._move_nbr else pos
                recipient = winners[landed]
                require(sim.workers[recipient].player == beneficiary, "collision handoff beneficiary")
                gains[recipient] += value
        root_holder_before = holder
        if tagged and holder in lost:
            p, load, beneficiary, winner = lost[holder]
            require(p == seat and tagged <= load + 1e-5, "tag exceeds carrier's lost cargo")
            if winner == seat:
                p0 = pre[holder][0]
                cmd = joint[seat].get(holder)
                landing = sim._move_nbr[cmd][p0] if cmd in sim._move_nbr else p0
                holder = winners[landing]
            else:
                outcome = (
                    "our_landmark_cash"
                    if beneficiary == seat
                    else (
                        "opponent_collision_capture"
                        if beneficiary is not None
                        else "lost_to_ground_or_overflow"
                    )
                )
                settled[outcome] += tagged
                tagged, holder = 0.0, None
        if tagged and holder in event["payments"]:
            side, amount = event["payments"][holder]
            require(tagged <= amount + 1e-5, "tag exceeds carrier's delivered cargo")
            settled["own_base_cash" if side == seat else "paid_to_enemy_base"] += tagged
            tagged, holder = 0.0, None
        if tagged:
            require(holder in sim.workers, "unclassified loaded carrier removal")
            fraction = event["ret"].get(holder, 1.0)
            settled["tree_tax_to_enemy"] += tagged * (1 - fraction)
            tagged *= fraction
            require(tagged <= sim.workers[holder].carbon + 1e-5, "tag exceeds surviving carrier's inventory")
        equal(tagged + sum(settled.values()), at_root, "root cargo conservation")
        for worker, w in sim.workers.items():
            if not w.is_collector:
                continue
            unloaded = event["payments"].get(worker, (None, 0.0))[1]
            tax = sim.cash.reductions.get(worker, 0.0)
            fresh = w.carbon + tax + unloaded - pre.get(worker, (None, 0.0, None))[1] - gains[worker]
            require(fresh >= -1e-5, "negative newly mined cargo")
            if joint[w.player].get(worker) in sim._move_nbr:
                equal(fresh, 0.0, "moving worker marked as harvesting")
            cumulative[f"seat_{w.player}_fresh_harvest"] += max(0.0, fresh)
            if worker == uid:
                cumulative["courier_fresh_harvest"] += max(0.0, fresh)
        for worker, (side, amount) in event["payments"].items():
            if worker in pre and pre[worker][2] == side:
                cumulative[f"seat_{side}_own_base_delivery"] += amount
                if worker == uid:
                    cumulative["courier_gross_own_base_delivery"] += amount
        hit = lost.get(uid)
        if hit:
            cumulative["courier_cargo_at_collision"] += hit[1]
            if hit[2] == 1 - seat:
                cumulative["courier_cargo_captured_by_enemy"] += hit[1]
        trace.append(
            dict(
                after_step=step + 1,
                courier_alive=uid in sim.workers,
                carrier_before=root_holder_before,
                carrier_after=holder,
                original_lot_still_carried=round(tagged, 6),
                cash=list(sim.cash),
                settled={k: round(v, 6) for k, v in settled.items()},
            )
        )
        receipt = dict(
            cut_at=step + 1,
            actual_transitions=i + 1,
            root_load=at_root,
            root_lot_retained=tagged,
            root_lot_own_base=settled["own_base_cash"],
            root_lot_own_landmark=settled["our_landmark_cash"],
            root_lot_enemy_collision=settled["opponent_collision_capture"],
            root_lot_enemy_tree_tax=settled["tree_tax_to_enemy"],
            root_lot_ground=settled["lost_to_ground_or_overflow"],
            root_lot_enemy_delivery=settled["paid_to_enemy_base"],
            courier_gross_own_base=cumulative["courier_gross_own_base_delivery"],
            courier_fresh_harvest=cumulative["courier_fresh_harvest"],
            courier_collision_cargo=cumulative["courier_cargo_at_collision"],
            courier_collision_paid_enemy=cumulative["courier_cargo_captured_by_enemy"],
            seat_own_base=[cumulative[f"seat_{s}_own_base_delivery"] for s in (0, 1)],
            seat_fresh_harvest=[cumulative[f"seat_{s}_fresh_harvest"] for s in (0, 1)],
            cash=list(sim.cash),
            seat_cash_events=[dict(c) for c in team],
            courier_alive=uid in sim.workers,
        )
        if i == 9 or (env.done and i < 9):
            checkpoints["first10"] = receipt
        if env.done:
            checkpoints["terminal"] = receipt
    scores = [p.reward for p in env.states]
    require(
        env.done and [p.status for p in env.states] == ["DONE", "DONE"] and scores == record["score"],
        "terminal score differs",
    )
    equal(scores[seat] - scores[1 - seat], record["margin"], "official margin")
    retain(wd / f"lot_trace_{arm}.json.gz", trace)
    return dict(
        **checkpoints,
        score_by_physical_seat=scores,
        margin=record["margin"],
        u=utility(scores, seat),
        wdl="W" if scores[seat] > scores[1 - seat] else "D" if scores[seat] == scores[1 - seat] else "L",
        search_calls=record["search_calls"],
        search_completed=record["search_completed"],
        search_changed=record["search_changed"],
        search_aborted=record["search_aborted"],
        search_single=record["search_single"],
        elapsed_seconds=record["elapsed"],
        validated_v1_steps=len(full),
        trace_sha256=sha(wd / f"lot_trace_{arm}.json.gz"),
        action_sha256=sha(action_path),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "parent", "alternative", "verify"])
    p.add_argument("--plan", type=Path, default=HERE / "branch_plan.json")
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--index", type=int, required=True)
    args = p.parse_args()
    r, case, raw, wd = register(args.plan, args.plan_sha256, args.index)
    if args.mode == "prepare":
        initialize(wd, r, case, raw)
        return
    require(
        read(wd / "live_protocol.json") == case and (wd / "live_shared_init.json").is_file(),
        "run prepare first",
    )
    with lock(wd / f".{args.mode}.lock"):
        if args.mode in ("parent", "alternative"):
            cpus = sorted(os.sched_getaffinity(0))
            require(len(cpus) == 1, "pin the native branch to its allocated CPU")
            require(
                all(
                    os.environ.get(k) == "1"
                    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
                ),
                "set all three math libraries to one thread",
            )
            path = wd / f"resources_{args.mode}.json"
            if not path.exists():
                retain(
                    path,
                    dict(
                        cpus=cpus,
                        cgroup_cpu_max=Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
                        cgroup_stat_at_start=Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines(),
                    ),
                )
            live.run(args.mode, wd, case, resume=True)
            return
        arms = {arm: summarize_arm(wd, r, raw, arm) for arm in ("parent", "alternative")}
        a, b = arms["parent"], arms["alternative"]
        diff = {
            cut: {
                k: b[cut][k] - a[cut][k]
                for k in (
                    "root_lot_own_base",
                    "root_lot_own_landmark",
                    "root_lot_enemy_collision",
                    "root_lot_enemy_tree_tax",
                    "root_lot_ground",
                    "courier_gross_own_base",
                    "courier_fresh_harvest",
                    "courier_collision_paid_enemy",
                )
            }
            for cut in ("first10", "terminal")
        }
        retain(
            wd / "comparison.json",
            dict(
                root=r,
                plan_sha256=args.plan_sha256,
                shared_init_sha256=sha(wd / "live_shared_init.json"),
                arms=arms,
                delta_alternative_minus_parent={
                    **diff,
                    "margin": b["margin"] - a["margin"],
                    "u": b["u"] - a["u"],
                },
                scope="One registered root-only order. Both policies respond anew thereafter. Terminal residual cargo is unsold. Root lot follows friendly collector handoffs, is written off upon loss to neutral ground or the opposing side, and is credited only upon observed settlement.",
            ),
        )


if __name__ == "__main__":
    main()
