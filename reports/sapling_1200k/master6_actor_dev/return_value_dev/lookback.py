"""Audit archived parent trajectories on the four named discovery maps, from own-seat public state."""

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import gzip
import hashlib
import json
import math
import sys

HERE = Path(__file__).resolve().parent
TRIAL = HERE.parent / "tree_response_training"
sys.path.insert(0, str(TRIAL / "planter_survival"))
import extract

sys.path.insert(0, str(HERE.parent / "return_denial_training"))
from policy import MOVES, _DEST, _DIST, _commitments

SEEDS = {621471120, 621471121, 621473120, 621473121}
PACK = "9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0"
LEADS = (0, 3, 5, 10)


def load_sources():
    records = []
    for directory in (TRIAL, TRIAL / "batches/intake_03"):
        receipt = extract.load(directory / "source_receipt.json")
        official = directory / "source/official_replay.json"
        assert extract.sha(official) == receipt["official_sha256"]
        signed = extract.load(official)["runs"]["source"]
        assert signed["pack_sha256"] == PACK
        paired = {(r["seed"], r["seat"]): r for r in signed["games"]}
        for g in receipt["games"]:
            sig = paired[g["seed"], g["seat"]]
            assert sig["statuses"] == ["DONE", "DONE"]
            assert sig["official_reward_by_seat"] == g["scores"]
            assert sig["transcript_sha256"] == g["raw_sha256"] and sig["map_sha256"] == g["map_sha256"]
            records.append(g)
    assert {(g["seed"], g["seat"]) for g in records} == {(s, p) for s in SEEDS for p in (0, 1)}
    return records


def snapshot(obs, own_commands, uid, cfg):
    p = obs["player"]
    own, enemy = obs["players"][p], obs["players"][1 - p]
    if uid not in own[2]:
        return None
    pos, load, role = own[2][uid]
    if role != "COLLECTOR":
        raise ValueError("carrier role changed")
    homes = list(own[1].values())
    other_homes = set(enemy[1].values())
    ours = min(_DIST[pos][h] for h in homes)
    enemy_trees = [cell for cell, age in enemy[3].values() if age < cfg["treeLifespan"]]
    _, commitments = _commitments(own, enemy, own_commands, cfg)
    cells = _DEST[pos]
    act = own_commands.get(uid) or "NONE"
    ranks = {}
    for ix, order in enumerate(MOVES):
        cell = cells[ix]
        my_load = load * (1 - cfg["moveCost"]) if ix else load
        # Test enemy commands separately: a lighter/equal enemy can also choose to stay.
        possible = []
        for wid, (e, c, k) in enemy[2].items():
            if k != "COLLECTOR":
                continue
            if c <= my_load:
                possible.append(_DIST[e][cell])
            elif e != cell and c * (1 - cfg["moveCost"]) <= my_load:
                possible.append(_DIST[e][cell])
        d = min(possible, default=99)
        conflict = any(other != uid and loc == cell for other, loc, _, _ in commitments)
        rank = dict(
            cell=cell,
            home=min(_DIST[cell][h] for h in homes),
            light_enemy_distance=d,
            enemy_eligible_permitted_to_contact_in_one=d <= 1,
            friendly_committed=conflict,
            other_home=cell in other_homes,
            enemy_tree_tax_neighborhood=sum(
                max(
                    min(abs(t % 15 - cell % 15), 15 - abs(t % 15 - cell % 15)),
                    min(abs(t // 15 - cell // 15), 15 - abs(t // 15 - cell // 15)),
                )
                <= 1
                for t in enemy_trees
            ),
        )
        ranks[order] = rank
    usual = ranks[act]
    alts = {
        a: r for a, r in ranks.items() if a != act and not r["friendly_committed"] and not r["other_home"]
    }
    free = {a: r for a, r in alts.items() if r["light_enemy_distance"] > 1}
    safe_equal_or_closer = [a for a, r in free.items() if r["home"] <= usual["home"]]
    # A risk-only alternative may delay return by one step; separately account for these.
    no_extra_or_one = [a for a, r in free.items() if r["home"] <= usual["home"] + 1]
    ncollectors = sum(r == "COLLECTOR" for _, _, r in own[2].values())
    # Visible at the collector's cell. Approximation uses the pre-step headcount, assumes
    # no collision, expiry/seizure, or collection-lowering recruit before the stay phase.
    occupied = (
        set(homes)
        | other_homes
        | {c for pl in (own, enemy) for c, age in pl[3].values() if age < cfg["treeLifespan"]}
    )
    gross = (
        0
        if pos in occupied
        else obs["carbon"][pos]
        * max(0.0, cfg["initialCollectRate"] - ncollectors * cfg["collectDecreaseRate"])
    )
    return dict(
        t=obs["step"],
        pos=pos,
        cargo=round(load, 4),
        cash=own[0],
        enemy_cash=enemy[0],
        home=ours,
        old=act,
        visible_stay_gross_estimate=round(gross, 4),
        old_risk=usual["light_enemy_distance"] <= 1,
        old_home=usual["home"],
        old_enemy_distance=usual["light_enemy_distance"],
        available=len(alts),
        same_or_closer=safe_equal_or_closer,
        at_most_one_further=no_extra_or_one,
        all_orders=ranks,
    )


def neighbor9(a, b):
    dx = abs(a % 15 - b % 15)
    dy = abs(a // 15 - b // 15)
    return min(dx, 15 - dx) <= 1 and min(dy, 15 - dy) <= 1


def actual_tax(before, after, events, cfg, seat):
    # Newly planted trees do not absorb on the planting step. An existing seized tree
    # immediately pays its new owner. The full-board view is strictly label-side.
    before_ids = {
        tid for pl in before["players"] for tid, (_, age) in pl[3].items() if age < cfg["treeLifespan"]
    }
    enemy = 1 - seat
    trees = [cell for tid, (cell, age) in after["players"][enemy][3].items() if tid in before_ids]
    rate = cfg["collectorAbsorptionRate"]
    taxes = {}
    for u, (pos, cargo, kind) in after["players"][seat][2].items():
        if kind != "COLLECTOR":
            continue
        frac = rate * sum(neighbor9(pos, c) for c in trees)
        if frac >= 1:
            raise ValueError("tags impossible to reconstruct once entire balance is depleted")
        tax = cargo * frac / (1 - frac)
        if tax:
            taxes[u] = tax
    official = sum(e[5] for e in events if e[0] == "tree_absorption" and e[1] == enemy)
    if not math.isclose(sum(taxes.values()), official, abs_tol=1e-6):
        raise ValueError("per-carrier tax and actual tree receipts diverged")
    return taxes


def shortest_tax_options(obs, uid, cfg, orders, facts):
    baseline = facts["old"]
    home = facts["home"]
    if baseline == "NONE" or home < 2 or facts["old_home"] != home - 1:
        return None
    own, enemy = obs["players"][obs["player"]], obs["players"][1 - obs["player"]]
    homes = list(own[1].values())
    trees = [(c, cfg["treeLifespan"] - age) for c, age in enemy[3].values() if age < cfg["treeLifespan"]]
    opponents = set(enemy[1].values())

    # These continuations are static, with the current enemy trees ageing and no new plants,
    # seizures, captures, harvesting or opponent movement. Arrival unloads before absorption.
    @lru_cache(None)
    def retention(cell, d):
        return (
            (1 - cfg["collectorAbsorptionRate"] * sum(neighbor9(cell, p) and life >= d for p, life in trees))
            if cell not in homes
            else 1.0
        )

    @lru_cache(None)
    def best(cell, d):
        if cell in homes:
            return 1.0
        rem = min(_DIST[cell][p] for p in homes)
        tail = [
            q for q in _DEST[cell][1:] if q not in opponents and min(_DIST[q][p] for p in homes) == rem - 1
        ]
        return retention(cell, d) * max((best(q, d + 1) for q in tail), default=0.0)

    usual = facts["all_orders"][baseline]
    if usual["light_enemy_distance"] <= 1 or usual["friendly_committed"] or usual["other_home"]:
        return None
    options = []
    for action, item in facts["all_orders"].items():
        if (
            item["home"] != home - 1
            or item["friendly_committed"]
            or item["other_home"]
            or item["light_enemy_distance"] <= 1
        ):
            continue
        q = item["cell"]
        options.append(
            dict(action=action, cell=q, first_tax_fraction=1 - retention(q, 1), best_path_retained=best(q, 1))
        )
    if len(options) < 2:
        return None
    old = next(o for o in options if o["action"] == baseline)
    alt = max(
        (o for o in options if o["action"] != baseline),
        key=lambda o: (-o["first_tax_fraction"], o["best_path_retained"], -MOVES.index(o["action"])),
    )
    return dict(old=old, alt=alt, initial_home=home, available=len(options))


def game(g):
    path = extract.ROOT / g["raw_path"]
    assert extract.sha(path) == g["raw_sha256"]
    raw = extract.load(path)
    cfg, seat = raw["game_configuration"], g["seat"]
    sim = extract.FastCarbon(cfg).reset(seed=g["seed"])
    if hashlib.sha256(bytes(int(x) for x in sim.carbon)).hexdigest() != g["map_sha256"]:
        raise ValueError("map changed")
    assert sim.config["moveCost"] == sim.config["smeltCost"] == 0
    cfg = sim.config
    assert sim.size == 15
    frames, events, buckets = [], [], defaultdict(Counter)
    taxlog = Counter()
    alternatives = []
    tax_cases = []
    projected_choices = []
    active = []

    def event(t, u, typ, v, details):
        buckets[typ]["count_all_amounts"] += 1
        buckets[typ]["amount_all"] += v
        if v < 100:
            return
        buckets[typ]["count_ge100"] += 1
        buckets[typ]["amount_ge100"] += v
        ev = dict(t=t, uid=u, kind=typ, amount=v, details=details, lookback={})
        for k in LEADS:
            j = t - k
            if not 0 <= j < len(frames):
                continue
            obs, act = frames[j]
            state = snapshot(obs, act, u, cfg)
            if state is not None:
                ev["lookback"][str(k)] = state
        events.append(ev)

    for t, orders in enumerate(raw["actions"]):
        pre = sim.obs(seat)
        frames.append((pre, orders[seat]))
        before = sim.full_observation()
        eligible = {}
        projected = {}
        for u, (cell, load, kind) in pre["players"][seat][2].items():
            if kind != "COLLECTOR" or load < 100:
                continue
            taxlog["heavy_worker_turns"] += 1
            act = orders[seat].get(u) or "NONE"
            if act == "NONE":
                continue
            facts = snapshot(pre, orders[seat], u, cfg)
            option = shortest_tax_options(pre, u, cfg, orders[seat], facts)
            if option:
                taxlog["two_safe_shortest_options"] += 1
                old, alt = option["old"], option["alt"]
                if (
                    alt["best_path_retained"] > old["best_path_retained"] + 1e-9
                    and alt["first_tax_fraction"] <= old["first_tax_fraction"] + 1e-9
                ):
                    taxlog["later_strictly_less_tax_static"] += 1
                    taxlog["later_static_root_cash_gain_sum_over_overlapping_roots"] += (
                        alt["best_path_retained"] - old["best_path_retained"]
                    ) * load
                    projected[u] = (load, option)
                if old["first_tax_fraction"] > alt["first_tax_fraction"] + 1e-9:
                    taxlog["less_tax_first_step"] += 1
                    if alt["best_path_retained"] + 1e-9 >= old["best_path_retained"]:
                        eligible[u] = (load, option)
        for u, (load, opt) in projected.items():
            record = dict(
                t=t,
                uid=u,
                cargo=load,
                **opt,
                factual=dict(
                    total_tax=0.0,
                    root_principal_tax=0.0,
                    new_harvest=0.0,
                    incoming_collision_cargo=0.0,
                    root_principal=load,
                    trace=[],
                ),
            )
            projected_choices.append(record)
            active.append(record)
        saved = sim.get_state() if eligible else None
        after, cash = extract.step(sim, before, cfg, orders)
        taxes = actual_tax(before, after, cash, cfg, seat)
        for r in active[:]:
            u = r["uid"]
            f = r["factual"]
            pos, prevload, _ = pre["players"][seat][2][u]
            cell = _DEST[pos][MOVES.index(orders[seat].get(u) or "NONE")]
            coll = next((z for z in sim.last_collisions if z[0] == u), None)
            ending = after["players"][seat][2].get(u)
            delivery = next(
                (
                    e
                    for e in cash
                    if e[0] in ("own_delivery", "opponent_worker_delivery_to_this_base") and e[3] == u
                ),
                None,
            )
            if coll:
                f.update(
                    fate="same_worker_collision_death",
                    end_t=t,
                    gross_own_delivery=0.0,
                    tagged_own_delivery=0.0,
                )
                active.remove(r)
                continue
            if not ending:
                raise ValueError("unexpected live heavy worker disappeared")
            incoming = 0.0
            for dead, side, amount, recipient, win in sim.last_collisions:
                if win != seat:
                    continue
                v = before["players"][side][2].get(dead)
                if not v:
                    continue
                arrival = _DEST[v[0]][MOVES.index(orders[side].get(dead) or "NONE")]
                if arrival == cell:
                    incoming += amount
            tax = taxes.get(u, 0.0)
            paid = delivery[2] if delivery else 0.0
            new = ending[1] + tax + paid - prevload - incoming
            if new < -1e-6:
                raise ValueError("negative inferred harvest")
            f["new_harvest"] += max(0.0, new)
            f["total_tax"] += tax
            f["incoming_collision_cargo"] += incoming
            f["trace"].append(
                dict(
                    t=t,
                    a=orders[seat].get(u) or "NONE",
                    cell=cell,
                    cargo_start=prevload,
                    new_harvest=max(0.0, new),
                    won_or_recovered=incoming,
                    tax=tax,
                    credited=paid,
                )
            )
            if delivery:
                f.update(
                    fate="own_delivery" if delivery[1] == seat else "enemy_base_deposit",
                    end_t=t,
                    gross_own_delivery=paid if delivery[1] == seat else 0.0,
                    tagged_own_delivery=f["root_principal"] if delivery[1] == seat else 0.0,
                )
                active.remove(r)
                continue
            if tax:
                tax_of_principal = f["root_principal"] * tax / (ending[1] + tax)
                f["root_principal_tax"] += tax_of_principal
                f["root_principal"] -= tax_of_principal
        for u, v in taxes.items():
            taxlog["all_worker_tax_cash_to_opponent"] += v
            original = pre["players"][seat][2].get(u)
            if original and original[1] >= 100:
                taxlog["root_heavy_tax_cash_to_opponent"] += v
                taxlog["root_heavy_actual_tax_worker_turns"] += 1
                existing = snapshot(pre, orders[seat], u, cfg)
                tax_cases.append(
                    dict(
                        t=t,
                        uid=u,
                        root_cargo=original[1],
                        actual_tax=v,
                        old=existing["old"],
                        root_home=existing["home"],
                        old_home=existing["old_home"],
                        old_risk=existing["old_risk"],
                        old_enemy_trees=existing["all_orders"][existing["old"]][
                            "enemy_tree_tax_neighborhood"
                        ],
                        safe_same_distance_and_lower_tree_exposure=[
                            a
                            for a, r in existing["all_orders"].items()
                            if not r["friendly_committed"]
                            and not r["other_home"]
                            and r["light_enemy_distance"] > 1
                            and r["home"] <= existing["old_home"]
                            and r["enemy_tree_tax_neighborhood"]
                            < existing["all_orders"][existing["old"]]["enemy_tree_tax_neighborhood"]
                        ],
                        visible_stay_gross_estimate=existing["visible_stay_gross_estimate"],
                    )
                )
        for u, (load, opts) in eligible.items():
            altorders = [dict(z) for z in orders]
            altorders[seat][u] = opts["alt"]["action"]
            alt_sim = extract.FastCarbon(cfg).set_state(saved)
            altstate, altcash = extract.step(alt_sim, before, cfg, altorders)
            alt_taxes = actual_tax(before, altstate, altcash, cfg, seat)
            actual = taxes.get(u, 0.0)
            alternate = alt_taxes.get(u, 0.0)
            own_cargo = after["players"][seat][2].get(u)
            alt_cargo = altstate["players"][seat][2].get(u)
            if not own_cargo or not alt_cargo:
                raise ValueError("one-step-safe option lost the carrier")
            alternatives.append(
                dict(
                    t=t,
                    uid=u,
                    root_cargo=load,
                    old=opts["old"],
                    alt=opts["alt"],
                    home=opts["initial_home"],
                    direct_tax_original=actual,
                    direct_tax_alternative=alternate,
                    carrier_stock_change=alt_cargo[1] - own_cargo[1],
                    own_cash_change=altstate["players"][seat][0] - after["players"][seat][0],
                    enemy_cash_change=altstate["players"][1 - seat][0] - after["players"][1 - seat][0],
                )
            )
        direct = {e[3]: e[0] for e in cash if e[0] in ("tree_collision_cash", "base_collision_cash")}
        for u, p, v, recipient, winner in sim.last_collisions:
            w = pre["players"][seat][2].get(u)
            if p != seat or w is None or w[2] != "COLLECTOR" or v <= 0:
                continue
            if recipient == seat:
                typ = "friendly_transfer"
            elif recipient is None:
                typ = "ground_or_overflow"
            elif winner is not None:
                typ = "enemy_worker"
            else:
                typ = "enemy_" + direct[u]
            event(t, u, typ, v, {"recipient": recipient, "surviving_collector_seat": winner})
        for row in cash:
            typ, credited, v, *detail = row
            if typ not in ("opponent_worker_delivery_to_this_base", "own_delivery") or v <= 0:
                continue
            u, cell = detail
            if u not in pre["players"][seat][2]:
                continue
            tag = "own_delivery" if credited == seat else "enemy_base_deposit"
            event(t, u, tag, v, dict(cell=cell, credited=credited))
    for r in active:
        r["factual"].update(
            fate="ended_carrying", end_t=sim.step_count, gross_own_delivery=0.0, tagged_own_delivery=0.0
        )
    for u, (cell, v, k) in sim.full_observation()["players"][seat][2].items():
        if k == "COLLECTOR" and v > 0:
            event(sim.step_count, u, "undelivered_terminal_stock", v, {"cell": cell})
    assert sim.step_count == g["steps"] and sim.cash == g["scores"]
    return dict(
        seed=g["seed"],
        seat=seat,
        map_sha256=g["map_sha256"],
        raw_sha256=g["raw_sha256"],
        steps=sim.step_count,
        official_scores=g["scores"],
        buckets=buckets,
        events=events,
        tree_tax=taxlog,
        heavy_tax_transfers=tax_cases,
        equal_route_comparisons=alternatives,
        shortest_static_forward_only_choices=projected_choices,
    )


BAD = {
    "enemy_worker",
    "enemy_tree_collision_cash",
    "enemy_base_collision_cash",
    "enemy_base_deposit",
    "ground_or_overflow",
}


def comparison(games):
    matched = []
    for g in games:
        neg = [r for r in g["events"] if r["kind"] in BAD and "3" in r["lookback"]]
        pos = [r for r in g["events"] if r["kind"] == "own_delivery" and "3" in r["lookback"]]
        used = set()
        pairs = []
        for r in sorted(neg, key=lambda v: (-v["amount"], v["t"], v["uid"])):
            a = r["lookback"]["3"]
            eligible = []
            for j, s in enumerate(pos):
                b = s["lookback"]["3"]
                if j in used or min(a["cargo"], b["cargo"]) <= 0:
                    continue
                ratio = a["cargo"] / b["cargo"]
                dh = abs(a["home"] - b["home"])
                if not 0.5 <= ratio <= 2 or dh > 2:
                    continue
                eligible.append((abs(math.log(ratio)) + dh / 3, abs(r["t"] - s["t"]), j))
            if not eligible:
                continue
            j = min(eligible)[2]
            used.add(j)
            s = pos[j]
            b = s["lookback"]["3"]
            pairs.append(
                dict(
                    loss=[r["t"], r["uid"], r["amount"], r["kind"]],
                    delivery=[s["t"], s["uid"], s["amount"]],
                    loss_cargo_home=[a["cargo"], a["home"]],
                    delivery_cargo_home=[b["cargo"], b["home"]],
                    loss_risk=a["old_risk"],
                    delivery_risk=b["old_risk"],
                    loss_safe_equal_or_closer=bool(a["same_or_closer"]),
                    delivery_safe_equal_or_closer=bool(b["same_or_closer"]),
                )
            )
        matched.append(dict(seed=g["seed"], seat=g["seat"], all_loss_with_3step_worker=len(neg), pairs=pairs))
    return matched


def stats(events, kind, k):
    chosen = [e for e in events if e["kind"] in kind]
    x = [e["lookback"][str(k)] for e in chosen if str(k) in e["lookback"]]
    heavy = [r for r in x if r["cargo"] >= 100]
    risk = [r for r in heavy if r["old_risk"]]
    same = [r for r in risk if r["same_or_closer"]]
    plus = [r for r in risk if r["at_most_one_further"]]
    return [len(chosen), len(x), len(heavy), len(risk), len(same), len(plus)]


def output(games):
    ev = [e for g in games for e in g["events"]]
    groups = defaultdict(Counter)
    for g in games:
        for k, b in g["buckets"].items():
            groups[k].update(b)
    pairs = comparison(games)
    payload = dict(pack_sha256=PACK, scope="discovery only", lags=LEADS, matched_controls=pairs, games=games)
    extract.write(HERE / "lookback.json.gz", payload)
    out = [
        "# 重载工出事前，原版 v1 看得见什么",
        "",
        "仅查看 01、03 两批训练集的 4 张地图、8 场换席原版 Master3s2 行动带。逐拍重放 2,392 次，官方 Board 与 FastCarbon 的资金、实体和地块逐步一致。每笔事件以同一采集工的实际金额计。最后仍背着的货不进入现金，单列清仓机会。",
        "",
        "实际被敌工截走的 100 以上货物共 2 笔，金额 500.21。260 笔同规模正常交付共 45,896.59。3,397 个重货工拍中，12 拍实际向敌树付税 206.52。静态路线模型标出的 10 个节税根，在 v1 原轨迹中全部同工完整回库、途中实际付税均为零。",
        "",
        "| 事件 | 全金额笔数 | 全金额合计 | 单笔至少100的笔数 | 这些笔金额 |",
        "|---|---:|---:|---:|---:|",
    ]
    for typ, v in sorted(groups.items()):
        out.append(
            f"| {typ} | {v['count_all_amounts']} | {v['amount_all']:.2f} | {v['count_ge100']} | {v['amount_ge100']:.2f} |"
        )
    out += [
        "",
        "回看以事件发生的决策回合 t 为起点。终局库存以 t=299 计。每个较早根都重新检查工人身份、当时是否已有 100 货和真实 v1 命令。危险指更轻或等重敌采集工在一拍内可以到该落点。此指标不假设敌人一定进攻。可替代要求不与同拍己方既有命令或有效新招工冲突，并排除敌方基地。比较‘同距离或更近’时以原命令落点到最近己方库的距离为准。",
        "",
        "| 结果组 | 提前步 | 事件总数 | 同工尚在 | 当时已重载 | 原命令落点危险 | 且有安全不绕远替代 | 若容许多1格 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, kind in (
        ("被抢/掉地/投敌库", BAD),
        ("真实回库", {"own_delivery"}),
        ("终局背包", {"undelivered_terminal_stock"}),
    ):
        for k in LEADS:
            v = stats(ev, kind, k)
            out.append("| " + " | ".join(map(str, [name, k, *v])) + " |")
    pp = [p for record in pairs for p in record["pairs"]]
    out += [
        "",
        "提前三步的同场正反对照按实际物理席匹配。先取较大损失；成功回库工的当时载量须在损失工 0.5–2 倍、到库距离差≤2，在满足条件的样本中取载量和距离最接近者，每次成功只用一次。资产体量与距离仍可能不平衡。",
        "",
        f"共 {len(pp)} 对。原命令落点可能被轻敌工撞到，损失组 {sum(v['loss_risk'] for v in pp)}/{len(pp)}，成功组 {sum(v['delivery_risk'] for v in pp)}/{len(pp)}。兼具危险和同距或更短的安全替代，损失组 {sum(v['loss_risk'] and v['loss_safe_equal_or_closer'] for v in pp)}/{len(pp)}，成功组 {sum(v['delivery_risk'] and v['delivery_safe_equal_or_closer'] for v in pp)}/{len(pp)}。",
        "",
        "### 真实例子",
        "",
        "| 图 / 席 | 结果 / t | 金额 | t−3 工人 / 货 / 到库距离 | 原令 / 敌可达 | 同距或更近的安全方向 |",
        "|---|---|---:|---|---|---|",
    ]
    # Show one actual high-value loss and one real receipt with a potentially deceptive warning per game.
    for g in games:
        for cat in (BAD, {"own_delivery"}):
            pick = sorted(
                (
                    e
                    for e in g["events"]
                    if e["kind"] in cat and "3" in e["lookback"] and e["lookback"]["3"]["cargo"] >= 100
                ),
                key=lambda e: (
                    -int(e["lookback"]["3"]["old_risk"] and bool(e["lookback"]["3"]["same_or_closer"])),
                    -e["amount"],
                    e["t"],
                ),
            )
            if not pick:
                continue
            e = pick[0]
            a = e["lookback"]["3"]
            out += [
                f"| {g['seed']} / {g['seat']} | {e['kind']} / {e['t']} | {e['amount']:.2f} | {e['uid']} / {a['cargo']:.2f} / {a['home']} | {a['old']} / {int(a['old_risk'])} | {', '.join(a['same_or_closer']) or '无'} |"
            ]
    out += [
        "",
        "### 重载工被敌树逐拍吸走的收入",
        "",
        "以出令前本工载量至少100划界。敌树对本工的实收由结算后工号载量与当拍有效敌树反推；本拍所有工人的税合计逐步与敌方每棵树实记的敌货收入核对。相同工人的不同拍可重复出现，金额都是不同笔真实转账。",
        "",
        "| 图 / 席 | 重载工拍 | 其中被抽税工拍 | 重货实付敌树 | 全部工人实付敌树 | v1 正向返库且有另条同样一步安全的最短路线 | 另路本步少税、全程固定树留存不减 | 这些根当步实收税差 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    total = Counter()
    aa = []
    for g in games:
        row = g["tree_tax"]
        total.update(row)
        opts = g["equal_route_comparisons"]
        aa.extend(opts)
        saving = sum(x["direct_tax_original"] - x["direct_tax_alternative"] for x in opts)
        out.append(
            f"| {g['seed']} / {g['seat']} | {row['heavy_worker_turns']} | {row['root_heavy_actual_tax_worker_turns']} | {row['root_heavy_tax_cash_to_opponent']:.2f} | {row['all_worker_tax_cash_to_opponent']:.2f} | {row['two_safe_shortest_options']} | {len(opts)} | {saving:.2f} |"
        )
    out.append(
        f"| 合计 | {total['heavy_worker_turns']} | {total['root_heavy_actual_tax_worker_turns']} | {total['root_heavy_tax_cash_to_opponent']:.2f} | {total['all_worker_tax_cash_to_opponent']:.2f} | {total['two_safe_shortest_options']} | {len(aa)} | {sum(x['direct_tax_original']-x['direct_tax_alternative'] for x in aa):.2f} |"
    )
    out += [
        "",
        "双方现有工人与出令时公开敌树用于筛根。原令与替代均为移动、均使最近库距离缩短一格，均不入敌库、不与友军已承诺的落点冲突，也不进轻敌采集工下一拍可达格。比较同根两条最短路只在当拍敌树随龄消亡的假设下估计送达比例，未预测新种/易主或敌工今后的反应。右列用两引擎在同一当拍敌令下确定实收税差，只说明该步变化；重复经过相邻回合的节税不能叠加声称整个策略会多赚这些钱。移动两臂当拍均不采集，后续额外经营或抢货机会需双边按新局面续行才能定价。",
        "",
    ]
    out += [
        f"放宽到当前一步可同税、只要求全程某条等长路按现有敌树可更少纳税，出现 {total['later_strictly_less_tax_static']} 个工拍。静态债务差合计 {total['later_static_root_cash_gain_sum_over_overlapping_roots']:.2f}，含同工相邻回合，不是可兑现预测。",
        "",
        "| 图 / 席 / 决策拍 | 工 / 重货 | 实付敌树 | 原令 / 到库距离变动 | 安全不绕远且眼前少碰敌树的替代 | 原地采收毛估 |",
        "|---|---|---:|---|---|---:|",
    ]
    for g in games:
        for r in g["heavy_tax_transfers"]:
            out += [
                f"| {g['seed']} / {g['seat']} / {r['t']} | {r['uid']} / {r['root_cargo']:.2f} | {r['actual_tax']:.2f} | {r['old']} / {r['root_home']}→{r['old_home']} | {', '.join(r['safe_same_distance_and_lower_tree_exposure']) or '无'} | {r['visible_stay_gross_estimate']:.2f} |"
            ]
    out += [
        "",
        "对每个‘换首步在静态最短路上可能省税’的候选，用既有命令向前追同一工号，至第一次入库、死亡或终局。实际入库含路上新增采集与交战所得，原有货物的树税按当拍扣税前背包的本金占比归属。",
        "",
        "| 图 / 席 / t | 工 / 载货 | 原令→替代 | 静态理论差 | 实际首结算 / t | 当段本金实税 / 全税 | 期间新增采集 / 碰撞入货 | 真实正常入库 |",
        "|---|---|---|---:|---|---:|---:|---:|",
    ]
    for g in games:
        for r in g["shortest_static_forward_only_choices"]:
            f = r["factual"]
            gain = (r["alt"]["best_path_retained"] - r["old"]["best_path_retained"]) * r["cargo"]
            out += [
                f"| {g['seed']} / {g['seat']} / {r['t']} | {r['uid']} / {r['cargo']:.2f} | {r['old']['action']}→{r['alt']['action']} | {gain:.2f} | {f['fate']} / {f['end_t']} | {f['root_principal_tax']:.2f} / {f['total_tax']:.2f} | {f['new_harvest']:.2f} / {f['incoming_collision_cargo']:.2f} | {f['gross_own_delivery']:.2f} |"
            ]
    out += [
        "",
        "抢到背包与直接入账分列。掉地金额为工人携货离开其账户的毛额，可能部分因地块上限而销毁。税、后续抢回和他人再交付均不能重复算作自己的首次回库；真实交付金额可能包含同拍抢得的其他货物。终局库存有可能是最后几拍才获得，分母如实区分。停留继续采集的粗估值、对面树税邻域和所有候选方向保存在逐事件证据。该回看按未来事件分组，不是部署的触发率，也没有重走任何替代分支。",
        "",
        "[逐事件、各方向、逐席原始文件散列及配对表](lookback.json.gz) · [复算程序](lookback.py)",
    ]
    (HERE / "lookback.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    output([game(g) for g in load_sources()])
