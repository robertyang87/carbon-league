"""Audit saved original matches under a historical double-collector opening and a live split opening."""

from collections import Counter
from pathlib import Path
from statistics import mean, stdev
import argparse
import math
import os
import sys

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "blended_actor_arena"
sys.path.insert(0, str(OLD))
import scorecard
from shared import ROOT, read, require, save, sha

CHECKPOINTS = (1, 5, 10, 20, 50, 100, 200)
OBSERVED = ("hire_collector", "hire_planter", "plant", "seize", "own_delivery")


def intake(lots, seat, step):
    transfers = [lot for lot in lots if lot["acquirer_seat"] == seat and lot["post_t"] <= step]
    gross = sum(lot["cargo"] for lot in transfers)
    bank = sum(lot["immediate"].get("own_bank", 0.0) for lot in transfers)
    still = 0.0
    for lot in transfers:
        settled = sum(lot["immediate"].values())
        for when, outcome, worker, target, amount in lot["downstream"]:
            if when <= step:
                bank += amount if outcome == "own_bank" else 0.0
                # Terminal inventory remains on the board through the official cutoff.
                if outcome != "held_at_terminal":
                    settled += amount
        still += max(0.0, lot["cargo"] - settled)
    return {
        "enemy_worker_lots": len(transfers),
        "intercepted_cargo": gross,
        "intercepted_later_in_normal_bank": bank,
        "intercepted_still_in_friendly_custody": still,
    }


def prosperity(g):
    frames = g["_frames"]
    raw = read(ROOT / g["raw"])
    n = len(frames)
    seat = g["seat"]
    sim = scorecard.lots.FastCarbon(raw["game_configuration"]).reset(seed=g["seed"])
    for pair in raw["actions"]:
        sim.step(*pair)
    players = [f["pre_players"] for f in frames] + [sim.full_observation()["players"]]
    require([p[0] for p in players[-1]] == g["final_cash"], "postgame treasury")
    records = []
    for side in (0, 1):
        first = {}
        cash = Counter()
        count = Counter()
        snap = {}
        tree = None
        seat_opening = frames[0]["joint"][side]
        base_ids = sorted(players[0][side][1], key=lambda uid: int(uid.rsplit("-", 1)[1]))
        require(len(base_ids) == 2, "opening only valid under two real bases")
        open_table = []
        for uid in base_ids:
            cell = players[0][side][1][uid]
            birth = [
                (worker, attrs)
                for worker, attrs in players[1][side][2].items()
                if attrs[0] == cell and worker not in players[0][side][2]
            ]
            open_table.append(
                {
                    "base": uid,
                    "cell": cell,
                    "command": seat_opening.get(uid, "NONE"),
                    "surviving_newborns_after_action": [[worker, attrs[2]] for worker, attrs in birth],
                }
            )
        occupied_tree_steps = 0
        alive_planter_steps = 0
        first_plant_unit = None
        first_plant_uid = None
        first_try = None
        for f in frames:
            t = f["t"]
            turn = t + 1
            pre = players[t][side]
            post = players[turn][side]
            for e in f["cash_events"]:
                key, owner, money, *rest = e
                if owner != side:
                    continue
                cash[key] += money
                count[key] += 1
                if key in OBSERVED:
                    first.setdefault(
                        key,
                        {
                            "action_t": t,
                            "state_after": turn,
                            "amount": money,
                            "uid": rest[0],
                            "cell": rest[1],
                        },
                    )
                if key == "hire_planter" and first_try is None:
                    new = [
                        uid
                        for uid, (cell, v, role) in post[2].items()
                        if uid not in pre[2] and cell == rest[1] and role == "PLANTER"
                    ]
                    require(len(new) <= 1, "ambiguous newly hired planter")
                    first_try = dict(
                        action_t=t,
                        base=rest[0],
                        cell=rest[1],
                        survived_recruit_action=bool(new),
                        worker=new[0] if new else None,
                    )
                    first_plant_unit = first_try["worker"]
                if key == "plant" and first_plant_uid is None:
                    actual = [
                        uid for uid, (cell, age) in post[3].items() if cell == rest[1] and uid not in pre[3]
                    ]
                    require(len(actual) == 1, "new tree missing after payment")
                    first_plant_uid = actual[0]
                    tree = dict(
                        uid=first_plant_uid,
                        action_t=t,
                        builder=rest[0],
                        cell=rest[1],
                        purchase=-money,
                        first_holding_absorption=0.0,
                        first_holding_collision_credits=0.0,
                    )
                if tree and "exit_after" not in tree:
                    if key == "tree_absorption" and rest[0] == first_plant_uid:
                        tree["first_holding_absorption"] += money
                    if key == "tree_collision_cash" and rest[-1] == first_plant_uid:
                        tree["first_holding_collision_credits"] += money
            if (
                first_try
                and first_plant_unit
                and first_plant_unit not in post[2]
                and "last_alive_pre_action" not in first_try
            ):
                require(first_plant_unit in pre[2], "planter lifetime discontinuity")
                first_try["last_alive_pre_action"] = t
            if tree and "exit_after" not in tree and first_plant_uid not in post[3]:
                require(first_plant_uid in pre[3], "tree lifetime discontinuity")
                tree["exit_after"] = turn
                tree["exit_reason"] = (
                    "seized" if first_plant_uid in players[turn][1 - side][3] else "removed_by_rules"
                )
            worker_types = Counter(v[2] for v in post[2].values())
            occupied_tree_steps += len(post[3])
            alive_planter_steps += worker_types["PLANTER"]
            if turn in CHECKPOINTS or turn in (n - 20, n):
                lots = intake(g["capture_lots"], side, turn)
                require(
                    lots["intercepted_later_in_normal_bank"] <= cash["own_delivery"] + 1e-6,
                    "tagged bank exceeds total",
                )
                snap[str(turn)] = dict(
                    cash=post[0],
                    cash_by_operation=scorecard.clean(cash),
                    worker_count=dict(worker_types),
                    cumulative_operations=dict(count),
                    standing_trees=len(post[3]),
                    tree_owner_turns=occupied_tree_steps,
                    planter_worker_turns=alive_planter_steps,
                    living_cargo=sum(v[1] for v in post[2].values()),
                    tree_income=cash["tree_absorption"] + cash["tree_collision_cash"],
                    normal_deposits=cash["own_delivery"],
                    **lots,
                )
        if first_try:
            first_try["alive_at_terminal"] = first_plant_unit in players[n][side][2]
            first_try["live_postmove_states"] = sum(
                first_plant_unit in players[t][side][2] for t in range(1, n + 1)
            )
        if tree:
            tree["owned_at_terminal"] = first_plant_uid in players[n][side][3]
            last_state = tree.get("exit_after", n + 1) - 1
            tree["initial_holding_states"] = last_state - tree["action_t"]
            tree["first_holding_receipts_less_purchase"] = (
                tree["first_holding_absorption"] + tree["first_holding_collision_credits"] - tree["purchase"]
            )
        require(
            abs(snap[str(n)]["tree_income"] - g["accounts_physical"][side]["tree_income"]) < 1e-6,
            "tree account",
        )
        for a, b in (
            ("intercepted_cargo", "physical_enemy_to_collector"),
            ("intercepted_later_in_normal_bank", "stolen_then_normal_own_deposit"),
            ("intercepted_still_in_friendly_custody", "stolen_still_carried_at_terminal"),
        ):
            scorecard.same(snap[str(n)][a], g["accounts_physical"][side]["cargo"].get(b, 0))
        records.append(
            dict(
                base_opening=open_table,
                first_actual_events=first,
                first_planter_hire=first_try,
                first_planted_tree=tree,
                states_after_actions=snap,
            )
        )
    return records


def episode(folder, seed, seat, role, identity):
    g = scorecard.replay(folder, seed, seat, role, identity)
    g["economic_timeline_by_physical_seat"] = prosperity(g)
    return g


def canonical_opening(g, expected):
    side = g["seat"]
    r = g["economic_timeline_by_physical_seat"][side]["base_opening"]
    require([v["command"] for v in r] == expected, "focal action dictionary differs from intervention")
    require(
        [
            (
                v["surviving_newborns_after_action"][0][1]
                if len(v["surviving_newborns_after_action"]) == 1
                else None
            )
            for v in r
        ]
        == [cmd[3:] for cmd in expected],
        "first-action bodies different from actual recruitment",
    )


def run_historical():
    binding = read(OLD / "games/binding.json")
    sg = read(OLD / "scorecard.json.gz")
    checked = []
    for raw in sg["pairs"]:
        seed = raw["seed"]
        seat = raw["seat"]
        g = episode(OLD / "games/candidate", seed, seat, "historical", binding["candidate"])
        canonical_opening(g, ["RECCOLLECTOR", "RECCOLLECTOR"])
        prior = raw["arms"]["candidate"]
        require(
            g["accounts_physical"] == prior["accounts_physical"]
            and g["official_margin"] == prior["official_margin"]
            and g["raw_sha256"] == prior["raw_sha256"],
            "historical accounts",
        )
        checked.append(
            (
                seed,
                seat,
                g["official_margin"],
                g["economic_timeline_by_physical_seat"][seat]["states_after_actions"]["100"]["tree_income"],
            )
        )
    return checked


def collect():
    proof = read(HERE / "comparison.json")
    plan = read(HERE / "protocol.json")
    require(
        proof["historical_baseline"] and proof["new_game_count"] == 4 and proof["formal_games"] == 0,
        "incorrect cohort",
    )
    incomes = read(HERE / "income.json")
    require(
        len(incomes) == 8 and proof["income_sha256"] == sha(HERE / "income.json"), "income/comparison binding"
    )
    groups = {}
    for baseline in plan["historical_baseline"]:
        seed = baseline["seed"]
        seat = baseline["seat"]
        arch = baseline["files"]
        for kind, reference in arch.items():
            require(
                sha(ROOT / reference["path"]) == reference["sha256"], "registered baseline changed: " + kind
            )
        pair = {}
        for arm, folder, pack, ops in (
            (
                "parent",
                (ROOT / arch["raw"]["path"]).parent,
                plan["baseline_sha256"],
                ["RECCOLLECTOR", "RECCOLLECTOR"],
            ),
            (
                "candidate",
                HERE / "games/candidate",
                plan["candidate_pack_sha256"],
                ["RECCOLLECTOR", "RECPLANTER"],
            ),
        ):
            game = episode(folder, seed, seat, arm, pack)
            require(game["map"] == baseline["terrain"], "physical map differs")
            rows = [p["arms"][arm] for p in proof["pairs"] if (p["seed"], p["seat"]) == (seed, seat)]
            require(len(rows) == 1 and rows[0]["raw_sha256"] == game["raw_sha256"], "comparison row identity")
            value = rows[0].get("official_margin", rows[0].get("margin"))
            scorecard.same(value, game["official_margin"])
            (entry,) = [x for x in incomes if (x["seed"], x["seat"], x["arm"]) == (seed, seat, arm)]
            require(
                entry["raw_sha256"] == game["raw_sha256"]
                and entry["official_score_by_seat"] == game["official_scores"],
                "income row identity",
            )
            for i, bank in enumerate(entry["by_seat"]):
                for name, amount in bank.items():
                    scorecard.same(amount, game["accounts_physical"][i]["cash"].get(name, 0))
            canonical_opening(game, ops)
            pair[arm] = game
        common = min(pair["parent"]["turns"], pair["candidate"]["turns"])
        divergence = {
            str(s): next(
                (
                    t
                    for t in range(common)
                    if pair["parent"]["_frames"][t]["joint"][s] != pair["candidate"]["_frames"][t]["joint"][s]
                ),
                None,
            )
            for s in (0, 1)
        }
        p = scorecard.paired(pair)
        p["observed_policy_joint_command_first_difference_by_seat"] = divergence
        groups[seed, seat] = p
    require(len(groups) == 4 and {seat for _, seat in groups} == {0, 1}, "all fixed pairs required")
    terrains = {
        seed: mean(groups[seed, s]["terminal_margin_delta"] for s in (0, 1))
        for seed in sorted(set(seed for seed, _ in groups))
    }
    require(len(terrains) == 2, "two real boards")
    scorecard.same(mean(terrains.values()), proof["paired_delta"])
    return dict(
        historical_baseline=True,
        new_original_games=4,
        paired_seed_seats=4,
        formal_independent_games=0,
        scope="Same official starting maps and physical seat. Historical reference trajectories were already observed. Full examiner used its live wall clock and natural temperature draws. No common opponent RNG or equal opponent responses are claimed. Root opening execution is identified, subsequent policy paths are descriptive.",
        input_hashes={
            key: sha(path)
            for key, path in {
                "protocol": HERE / "protocol.json",
                "comparison": HERE / "comparison.json",
                "income": HERE / "income.json",
                "baseline_scorecard": OLD / "scorecard.json.gz",
                "cash_replay": OLD / "scorecard.py",
                "economy_source": Path(__file__),
            }.items()
        },
        paired_delta_by_map=terrains,
        paired_delta=mean(terrains.values()),
        cluster_se=stdev(terrains.values()) / math.sqrt(len(terrains)),
        pairs=list(groups.values()),
    )


def side_context(game, offset=0):
    return game["economic_timeline_by_physical_seat"][game["seat"] ^ offset]


def sample_state(g, offset, horizon):
    return side_context(g, offset)["states_after_actions"][
        str(g["turns"] if horizon == "terminal" else min(horizon, g["turns"]))
    ]


def report(item):
    pairs = item["pairs"]
    roles = ("parent", "candidate")
    games = {a: [p["arms"][a] for p in pairs] for a in roles}
    title = {"parent": "历史双采集工", "candidate": "新一采一植"}
    lines = ["# 首拍招工干预对原版 Master3s2 的四场开发比赛", ""]
    for a in roles:
        scores = Counter(g["result"] for g in games[a])
        msg = f"**{title[a]}**四场 {scores['win']} 胜、{scores['draw']} 平、{scores['loss']} 负，平均官方终局分差 {mean(g['official_margin'] for g in games[a]):+,.2f}。"
        lines.append(msg)
    mdeltas = list(item["paired_delta_by_map"].values())
    gain = item["paired_delta"]
    lines.extend(
        [
            "",
            f"两张地图各换席一次，四组差值的平均为 {gain:+,.2f}。按图先合并两席，分别为 {mdeltas[0]:+,.2f} 和 {mdeltas[1]:+,.2f}，地图聚类标准误 {item['cluster_se']:.2f}。四份双采集工回放来自已观察的开发场，本轮只新增四份一采一植回放。考官保持原版自然采样及真实墙钟，两组没有共享后续随机数。这四组不能认证正式考卷胜率或独立估计首拍的平均因果效应。",
            "",
            "| 地图种子 / 本方席位 | 历史双采分差 | 新分差 | 同图同席差 | 历史/新步数 | 考官首个实令不同的行动步 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for p in pairs:
        old, new = (p["arms"][a] for a in roles)
        step = p["observed_policy_joint_command_first_difference_by_seat"][str(1 - p["seat"])]
        lines.append(
            f"| {p['seed']} / {p['seat']} | {old['official_margin']:+.2f} | {new['official_margin']:+.2f} | {p['terminal_margin_delta']:+.2f} | {old['turns']}/{new['turns']} | {step if step is not None else '全同'} |"
        )
    shared_first = sum(
        p["observed_policy_joint_command_first_difference_by_seat"][str(1 - p["seat"])] != 0 for p in pairs
    )
    lines.extend(
        [
            "",
            f"四组成对首拍中，考官提交完全相同命令的有 {shared_first}/4 组。这些组的第一步招工差能在固定考官当步行动下逐一复算。第一步之后，两条轨迹均接受原版考官重新决策。",
        ]
    )
    lines.extend(
        [
            "",
            "## 首拍招工与第一棵树",
            "",
            "四组逐物理基地核验。两臂在 0 号基地实际招出采集工，1 号基地在历史场实际招采集工，在新场实际招种植工。表中首次招种植工和首次种树均取真实扣款的行动步 t，状态从 t+1 起计。种植工存续记录首次所招同一工人的实际在场步数，首棵树记录本方首次持有期，夺树后的其他持有期另记。",
            "",
            "| 地图 / 席 | 实施 | 首次招种植工 t | 该工在场步数 | 首次种树 t | 首棵树持有步数 | 首棵树首持实收减种价 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for p in pairs:
        for a in roles:
            c = side_context(p["arms"][a])
            w = c["first_planter_hire"]
            tr = c["first_planted_tree"]
            vals = [w["action_t"], w["live_postmove_states"]] if w else ["无", "—"]
            vals += (
                [
                    tr["action_t"],
                    tr["initial_holding_states"],
                    f"{tr['first_holding_receipts_less_purchase']:+.2f}",
                ]
                if tr
                else ["无", "—", "—"]
            )
            lines.append(f"| {p['seed']} / {p['seat']} | {title[a]} | " + " | ".join(map(str, vals)) + " |")
    lines.extend(
        [
            "",
            "## 双方收入、存续与实际截货",
            "",
            "每项是各组四场平均。活树和在场种植工取该状态的存量，树所得含树吸收和树格碰撞。抢得货进入在运本金，其中正常回库是已兑现的子集，终局在运未计入现金。",
            "",
            "| 时点 | 版本与收款方 | 活种植工 | 活树 | 正常回库 | 树所得 | 招工支出 | 种/夺树支出 | 实抢货量 | 其中已回库 | 其中仍在运 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for h in (20, 50, 100, "terminal"):
        for a in roles:
            for offset in (0, 1):
                arr = []
                for g in games[a]:
                    z = sample_state(g, offset, h)
                    cost = Counter(z["cash_by_operation"])
                    arr.append(
                        [
                            z["worker_count"].get("PLANTER", 0),
                            z["standing_trees"],
                            z["normal_deposits"],
                            z["tree_income"],
                            -cost["hire_collector"] - cost["hire_planter"],
                            -cost["plant"] - cost["seize"],
                            z["intercepted_cargo"],
                            z["intercepted_later_in_normal_bank"],
                            z["intercepted_still_in_friendly_custody"],
                        ]
                    )
                label = "终局" if h == "terminal" else f"第 {h} 步后"
                lines.append(
                    "| "
                    + label
                    + " | "
                    + title[a]
                    + ("本方" if not offset else "场考官")
                    + " | "
                    + " | ".join(f"{mean(col):,.2f}" for col in zip(*arr))
                    + " |"
                )

    def net(h, k):
        return mean(sample_state(g, 0, h)[k] - sample_state(g, 1, h)[k] for g in games["candidate"]) - mean(
            sample_state(g, 0, h)[k] - sample_state(g, 1, h)[k] for g in games["parent"]
        )

    lines.extend(
        [
            "",
            f"本方减考官，再取新方案减历史基线。100 步时树所得差变化 {net(100,'tree_income'):+.2f}，正常回库差变化 {net(100,'normal_deposits'):+.2f}。终局分别为 {net('terminal','tree_income'):+.2f} 与 {net('terminal','normal_deposits'):+.2f}。每个数只反映这两张已打开的开发图。",
            "",
            "[逐局收据](economy.json.gz)含两席首拍出令和新生 UID、第一种植工与树存续、分时账、真实截货本金去向、官方现金与 3/5/10 步转折行动。终局灭绝按官方得分单独结算。[冻结协议](protocol.json) · [官方终局](comparison.json) · [每局现金账](income.json)",
        ]
    )
    lines.extend(
        [
            "",
            "## 最后二十步的已结算树与运货收入",
            "",
            "| 版本与收款方 | 最后至多20步树所得 | 最后至多20步回库 | 最后至多20步种/夺树支出 | 终局尚存在树 | 终局工人携货 | 官方奖励减终局现金 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for a in roles:
        for off in (0, 1):
            values = []
            for g in games[a]:
                z = sample_state(g, off, "terminal")
                early = sample_state(g, off, g["turns"] - 20) if g["turns"] > 20 else {}
                old = Counter(early.get("cash_by_operation", {}))
                new = Counter(z["cash_by_operation"])
                s = g["seat"] ^ off
                values.append(
                    [
                        z["tree_income"] - early.get("tree_income", 0),
                        z["normal_deposits"] - early.get("normal_deposits", 0),
                        (old["plant"] + old["seize"]) - (new["plant"] + new["seize"]),
                        z["standing_trees"],
                        z["living_cargo"],
                        g["official_scores"][s] - g["final_cash"][s],
                    ]
                )
            lines.append(
                "| "
                + title[a]
                + ("本方" if not off else "场考官")
                + " | "
                + " | ".join(f"{mean(col):,.2f}" for col in zip(*values))
                + " |"
            )
    lines.extend(["", "末步按官方实际现金结算，上表把尚未交付的货、仍在场的树及官方灭绝调整逐项列出。"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-check", action="store_true")
    args = parser.parse_args()
    if args.historical_check:
        print(run_historical())
        return
    item = collect()
    body = report(item).encode()
    save(HERE / "economy.json.gz", item)
    target = HERE / "RESULTS.md"
    if target.exists():
        require(target.read_bytes() == body, "existing report differs")
        return
    tmp = target.with_name(target.name + f".{os.getpid()}.part")
    tmp.write_bytes(body)
    os.replace(tmp, target)


if __name__ == "__main__":
    main()
