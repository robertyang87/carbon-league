"""Independently reconcile the official complete branch, actions, deposits, and first 3/5/10 steps."""

from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
ROOT = BASE.parents[1]
HELP = BASE / "planter_routes_dev/endgame_responses"
REC = BASE / "root_recovery_branches"
HORIZON = BASE / "master6_actor_dev/freight_horizon"


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sys.path.insert(0, str(REC))
import live
from branches import begin, clean, destination, public

# The original official cash verifier imports its sibling by a bare module name.
_prev = sys.modules.get("select_roots")
sys.modules["select_roots"] = mod("original_verifier_selection", HELP / "select_roots.py")
try:
    ledger = mod("ordered_native_cash_verifier", HELP / "verify.py").ledger
finally:
    if _prev is None:
        del sys.modules["select_roots"]
    else:
        sys.modules["select_roots"] = _prev
sys.path.insert(0, str(HORIZON))
freight = mod("original_freight_tax_labels", HORIZON / "data.py")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes())


def check(condition, why):
    if not condition:
        raise ValueError(why)


def audit(raw, case, row, wd, arm, plan):
    r = case["root"]
    t = r["t"]
    s = row["seat"]
    opp = 1 - s
    u = row["own_uid"]
    enemy = row["target_uid"]
    a = read(wd / f"live_{arm}.json")
    tape = wd / a["action_tape"]
    tail = read(tape)
    check(
        a["action_sha256"] == sha(tape)
        and a["original_trajectory_sha256"] == row["source_sha256"]
        and a["protocol_sha256"] == sha(wd / "live_protocol.json")
        and a["shared_init_sha256"] == sha(wd / "live_shared_init.json")
        and a["search_calls"] == len(tail) - 1
        and a["source_sha256"] == plan["examiner_sha256"]
        and a["pack_sha256"] == plan["compared_parent_sha256"]
        and a["statuses"] == ["DONE", "DONE"],
        "invalid native binding",
    )
    check(clean(tail[0][opp]) == clean(raw["actions"][t][opp]), "different simultaneous root enemy order")
    funding = ledger(raw, r, tail)
    check(
        funding["score_by_physical_seat"] == a["score"]
        and abs(funding["u"] - a["u"]) < 1e-8
        and abs(funding["margin"] - a["margin"]) < 0.011
        and len(tail) + t == a["terminal_t"],
        "official score or length mismatch",
    )
    cfg = raw["game_configuration"]
    freight.check_cfg(cfg)
    sim = freight.FastCarbon(cfg).reset(seed=row["seed"])
    pack = ROOT / "data/weights/sapling_1200k/terminal_return_fit_v1.py"
    check(sha(pack) == plan["compared_parent_sha256"], "untested friendly actor")
    own = live.module(pack, "actual_early_lane_friendly")
    for i, j in enumerate(raw["actions"][:t]):
        if i == 0:
            check(clean(own.agent(sim.obs(s), cfg)) == clean(j[s]), "opening order")
        else:
            own._FEAT.observe(sim.obs(s))
        sim.step(*j)
    sim.cash = freight.FreightCash(sim.cash)
    history = []
    money = Counter()
    summaries = {}
    for j, joint in enumerate(tail):
        orders = clean(own.agent(sim.obs(s), cfg))
        if j == 0:
            check(
                orders == clean(raw["actions"][t][s]) and orders.get(u, "NONE") == row["parent_order"],
                "root baseline",
            )
            if arm == "alternative":
                if row["alternative_order"] == "NONE":
                    orders.pop(u, None)
                else:
                    orders[u] = row["alternative_order"]
        check(
            orders == clean(joint[s]), "friendly original action differs outside the registered modification"
        )
        if j >= 10:
            if j == 10:
                sim.cash = list(sim.cash)
            sim.step(*joint)
            continue
        pre = {uid: (w.player, w.pos, w.carbon) for uid, w in sim.workers.items()}
        moves = {
            uid: destination(pos, joint[side].get(uid) or "NONE", cfg["size"])
            for uid, (side, pos, _) in pre.items()
        }
        event = freight.add_step(sim, joint)
        landed = {w.pos: wid for wid, w in sim.workers.items() if w.is_collector}
        gained = Counter()
        for lost, side, value, beneficiary, win in sim.last_collisions:
            if value and win is not None:
                gained[landed[moves[lost]]] += value
            if lost == u:
                money["own_focal_collision_removed"] += value
                if beneficiary == opp:
                    money["own_focal_collision_to_enemy"] += value
            if lost == enemy and beneficiary == s:
                money["target_collision_to_ours"] += value
        for uid, w in sim.workers.items():
            if not w.is_collector:
                continue
            fresh = (
                w.carbon
                + sim.cash.reductions.get(uid, 0)
                + sim.cash.deposited.get(uid, 0)
                - pre.get(uid, (None, None, 0))[2]
                - gained[uid]
            )
            check(fresh > -0.00001, "negative mined cargo")
            if w.player == s:
                money["own_team_actual_harvest"] += max(0.0, fresh)
            if uid == u:
                money["own_focal_actual_harvest"] += max(0.0, fresh)
        for uid, (beneficiary, amount) in event["payments"].items():
            if uid not in pre or pre[uid][0] != beneficiary:
                continue
            if uid == enemy:
                money["target_banked"] += amount
            if uid == u:
                money["own_focal_banked"] += amount
            money["own_team_banked" if beneficiary == s else "enemy_team_banked"] += amount
        tracked = funding["path"][j]
        history.append(
            dict(
                after_step=t + j + 1,
                our_worker=tracked["our_target"],
                enemy_worker=tracked["enemy_target"],
                our_order=tracked["focal_order"],
                enemy_order=tracked["enemy_target_order"],
                both_landed_same_cell=(moves[u] == moves[enemy] if u in moves and enemy in moves else None),
                cash=tracked["cash"],
                own_worker_moved=(int(pre[u][1] != moves[u]) if u in moves else None),
            )
        )
        money["own_focal_move_cargo_cost"] += 0.0  # supported config checked moveCost == 0.
        for h in (3, 5, 10):
            if j + 1 == min(h, len(tail)):
                summaries[str(h)] = {
                    k: money[k]
                    for k in (
                        "target_banked",
                        "own_focal_banked",
                        "own_team_banked",
                        "enemy_team_banked",
                        "own_focal_actual_harvest",
                        "own_team_actual_harvest",
                        "own_focal_collision_removed",
                        "own_focal_collision_to_enemy",
                        "own_focal_move_cargo_cost",
                        "target_collision_to_ours",
                    )
                }
                summaries[str(h)].update(
                    actual_steps=j + 1,
                    our_at_end=tracked["our_target"],
                    target_at_end=tracked["enemy_target"],
                    same_cell_encounters=sum(v["both_landed_same_cell"] is True for v in history),
                )
                for side, field in [(s, "own_team_banked"), (opp, "enemy_team_banked")]:
                    cash = sum(
                        v["amount"]
                        for v in funding["deposits"]
                        if v["seat"] == side and v["kind"] == "own_delivery" and v["t"] <= t + j + 1
                    )
                    check(abs(cash - summaries[str(h)][field]) < 0.01, "official horizon receipts disagree")
    return (
        dict(
            score_by_physical_seat=a["score"],
            terminal_t=a["terminal_t"],
            u=funding["u"],
            margin=funding["margin"],
            wdl=funding["wdl"],
            statuses=a["statuses"],
            action_sha256=sha(tape),
            root_opponent_observation_sha256=a["root_opponent_observation_sha256"],
            horizons=summaries,
            first10=history,
            terminal_by_source=funding["gross"],
            all_actual_deposits=funding["deposits"],
            search_calls=a["search_calls"],
            search_completed=a["search_completed"],
            search_aborted=a["search_aborted"],
            source_sha256=a["source_sha256"],
            pack_sha256=a["pack_sha256"],
            friendly_actual_orders_checked=len(tail),
        ),
        tail,
    )


def verify_case(directory):
    wd = Path(directory).resolve()
    plan_path = HERE / "protocol.json"
    plan = read(plan_path)
    case = read(wd / "live_protocol.json")
    (row,) = [x for x in plan["roots"] if (x["seed"], x["seat"]) == (case["seed"], case["seat"])]
    r = case["root"]
    check(
        (r["t"], r["worker"], r["enemy_worker"], r["old"], r["alternative"])
        == (row["t"], row["own_uid"], row["target_uid"], row["parent_order"], row["alternative_order"]),
        "unregistered working root",
    )
    check(
        sha(Path(live.__file__)) == plan["live_responder_sha256"], "original native implementation mismatch"
    )
    source = ROOT / row["source"]
    check(sha(source) == row["source_sha256"], "original tape checksum")
    raw = read(source)
    official = read(source.parent / "official_replay.json")["runs"][source.parent.name]
    (match,) = [g for g in official["games"] if (g["seed"], g["seat"]) == (row["seed"], row["seat"])]
    raw["initial_terrain_u8_sha256"] = match["map_sha256"]
    arms = {}
    tails = {}
    for arm in ("parent", "alternative"):
        arms[arm], tails[arm] = audit(raw, case, row, wd, arm, plan)
    base, other = arms["parent"], arms["alternative"]
    check(
        base["root_opponent_observation_sha256"] == other["root_opponent_observation_sha256"], "unequal root"
    )
    s = row["seat"]
    divergence = [
        row["t"] + i
        for i, (a, b) in enumerate(zip(tails["parent"], tails["alternative"]))
        if clean(a[1 - s]) != clean(b[1 - s])
    ]
    metrics = {"delta_u": other["u"] - base["u"], "delta_margin": other["margin"] - base["margin"]}
    for h in (3, 5, 10):
        for k in (
            "target_banked",
            "own_team_banked",
            "enemy_team_banked",
            "own_focal_actual_harvest",
            "own_team_actual_harvest",
            "target_collision_to_ours",
        ):
            metrics[f"{k}{h}_delta"] = other["horizons"][str(h)][k] - base["horizons"][str(h)][k]
    result = dict(
        seed=row["seed"],
        seat=s,
        t=row["t"],
        protocol_sha256=sha(plan_path),
        source_sha256=sha(source),
        official_map_sha256=match["map_sha256"],
        verify_sha256=sha(Path(__file__)),
        arms=arms,
        metrics=metrics,
        opponent_orders_different_at=divergence,
        common_rng_sha256=sha(wd / "live_shared_init.json"),
    )
    dest = wd / "verified.json.gz"
    if dest.exists():
        check(read(dest) == result, "a prior signed verification differs")
    else:
        live.save(dest, result)
    return result


if __name__ == "__main__":
    for directory in sys.argv[1:]:
        print(verify_case(directory))
