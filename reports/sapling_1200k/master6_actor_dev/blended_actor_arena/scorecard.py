"""Check whole native games against the official interpreter, close both cash accounts and trace stolen inventory."""

import argparse
from collections import Counter
import math
import os
from pathlib import Path
from statistics import mean, stdev
import sys

from shared import HERE, ACTOR, BASE, ROOT, read, sha, save, require

sys.path.insert(0, str(ACTOR / "lot_settlement_dev"))
import lots

sys.path.insert(0, str(BASE))
import official_game_replay as official

TIMES = (1, 3, 5, 10)
ROLES = ("parent", "candidate")


def cash_lines(events):
    groups = [Counter(), Counter()]
    for name, side, amount, *_ in events:
        groups[side][name] += amount
    return groups


def add_money(a, b, sign=1):
    for k, v in b.items():
        a[k] += sign * v


def same(a, b):
    require(math.isclose(a, b, abs_tol=2e-6, rel_tol=1e-10), f"cash conservation {a} vs {b}")


def clean(counter):
    return {k: round(float(v), 7) for k, v in sorted(counter.items())}


def deltas(series):
    """Mark largest actual positive and negative changes, including the actions leading into their window."""
    ret = {}
    for h in TIMES:
        if len(series) <= h:
            continue
        diffs = [round(series[t + h] - series[t], 6) for t in range(len(series) - h)]
        extrema = {}
        for label, func, test in (("gain", max, lambda x: x > 0), ("loss", min, lambda x: x < 0)):
            v = func(diffs)
            if not test(v):
                continue
            start = diffs.index(v)
            end = start + h
            extrema[label] = {
                "start": start,
                "end": end,
                "change": v,
                "start_lead": series[start],
                "end_lead": series[end],
                "preceding_action_starts": {str(k): max(start - k, 0) for k in (3, 5, 10)},
            }
        ret[str(h)] = extrema
    return ret


def split_delta(games, from_t, to_t):
    """Keep both policies' two physical players, checking money before reporting their contributions."""
    byarm = {}
    for name, g in games.items():
        s = g["seat"]
        tot = Counter()
        for f in g["_frames"][from_t:to_t]:
            p = cash_lines(f["cash_events"])
            add_money(tot, p[s])
            add_money(tot, p[1 - s], -1)
        want = g["cash_lead"][to_t] - g["cash_lead"][from_t]
        same(sum(tot.values()), want)
        byarm[name] = clean(tot)
    return byarm


def fresh_lots(t, trans, amounts, pre, post, events, cfg):
    require(
        cfg["moveCost"] == cfg["smeltCost"] == 0,
        "cargo tracing requires verified zero movement and smelting fees",
    )
    prev_trees = {uid for pl in pre["players"] for uid in pl[3]}
    treasury = {e[3]: e for e in events if e[0] in ("own_delivery", "opponent_worker_delivery_to_this_base")}
    answer = []
    for victim, (edge, _, _) in trans.items():
        for kind, reason, _, hunter, frac in edge:
            if kind != "lost_to_enemy" or reason != "enemy_worker":
                continue
            s = int(hunter.split("-")[1])
            own = post["players"][s]
            cargo = amounts[victim] * frac
            require(int(victim.split("-")[1]) != s, "wrong acquisition identity")
            if hunter in treasury:
                e = treasury[hunter]
                dest = e[4]
                terminal = "own_bank" if e[0] == "own_delivery" else "enemy_base_deposit"
                survival = 0.0
                now = {terminal: cargo}
            else:
                dest = own[2][hunter][0]
                trees = [pos for uid, (pos, age) in post["players"][1 - s][3].items() if uid in prev_trees]

                def near(pos):
                    a = abs(pos // 15 - dest // 15)
                    b = abs(pos % 15 - dest % 15)
                    return min(a, 15 - a) <= 1 and min(b, 15 - b) <= 1

                tax = cargo * cfg["collectorAbsorptionRate"] * sum(near(pos) for pos in trees)
                require(0 <= tax < cargo, "tree tax outside feasible balance")
                survival = cargo - tax
                now = {"enemy_tree_tax": tax} if tax else {}
            answer.append(
                {
                    "action_t": t,
                    "post_t": t + 1,
                    "acquirer_seat": s,
                    "victim": victim,
                    "receiver": hunter,
                    "cell": dest,
                    "cargo": cargo,
                    "immediate": now,
                    "after_step": survival,
                }
            )
    return answer


def replay(folder, seed, seat, arm, pack):
    rec = read(folder / f"receipt_{seed}_{seat}.json")
    g, certificate = rec["game"], rec["official"]
    raw = folder / g["raw_file"]
    tape = read(raw)
    require(
        sha(raw) == g["raw_sha256"] == certificate["transcript_sha256"] and tape["pack_sha256"] == pack,
        "bad pack/raw binding",
    )
    require(
        tape["game_configuration"]["size"] == 15
        and tape["candidate_seat"] == seat
        and tape["seed"] == seed
        and not tape["errors"]
        and not tape["audit_messages"],
        "unclean raw",
    )
    require(official.check(folder, g) == certificate, "official interpreter score failed")
    cfg = tape["game_configuration"]
    sim = lots.FastCarbon(cfg).reset(seed=seed)
    require(list(sim.carbon) == tape["initial_full_carbon"], "wrong initial map")
    total = [Counter(), Counter()]
    routed = [Counter(), Counter()]
    tree_tax = [0.0, 0.0]
    start = list(sim.cash)
    cash_lead = [start[seat] - start[1 - seat]]
    frames = []
    values = {}
    trans = []
    received = []
    observed_payments = Counter()
    for t, joint in enumerate(tape["actions"]):
        before = sim.full_observation()
        visible = []
        for side in (0, 1):
            seen = set()
            for cell, _, kind in before["players"][side][2].values():
                seen.update((cell,) if kind == "COLLECTOR" else sim._nbr4[cell])
            view = sim.obs(side)["carbon"]
            visible.append([[cell, view[cell]] for cell in sorted(seen)])
        principal, edges, after = lots.one_step(sim, joint, None, verify=True)
        ev = sim.cash.events
        cash = cash_lines(ev)
        for s in (0, 1):
            same(sum(cash[s].values()), after["players"][s][0] - before["players"][s][0])
            add_money(total[s], cash[s])
        for k, s, num, *extra in ev:
            if k == "tree_absorption":
                tree_tax[s] += extra[2]
            if k == "own_delivery":
                observed_payments[t + 1, extra[0]] += num
        emit = fresh_lots(t, edges, principal, before, after, ev, cfg)
        received += emit
        for uid, amount in principal.items():
            values[(t, uid)] = amount
            for k, reason, _, dst, share in edges[uid][0]:
                if k == "relay" and reason == "friendly_worker":
                    routed[int(uid.split("-")[1])]["friendly_transfer"] += share * amount
        trans.append(edges)
        frames.append(
            {
                "t": t,
                "pre_players": before["players"],
                "visible_carbon_cells_by_seat": visible,
                "pre_tree_creator_and_last_absorption": before["trees"],
                "joint": joint,
                "cash_events": list(ev),
                "collisions": list(sim.last_collisions),
                "raided_lots": [[x["victim"], x["receiver"], x["cell"], x["cargo"]] for x in emit],
            }
        )
        cash_lead.append(after["players"][seat][0] - after["players"][1 - seat][0])
    n = len(frames)
    require(n == certificate["steps"] and list(sim.cash) == tape["last_post_cash"], "wrong terminal")
    for uid, v in lots.positive_collectors(sim.full_observation(), None).items():
        values[(n, uid)] = v
    tl = lots.Timeline(values, trans, n)
    tagged_deposits = Counter()
    for item in received:
        s = item["acquirer_seat"]
        q = item["cargo"]
        eff = item["after_step"]
        new = Counter(item["immediate"])
        dates = []
        if new.get("own_bank", 0):
            tagged_deposits[item["post_t"], item["receiver"]] += new["own_bank"]
        if eff > 0:
            for e in tl.events(item["post_t"], item["receiver"]):
                if e["is_terminal_allocation"]:
                    amount = eff * e["fraction"]
                    new[e["reason"]] += amount
                    if e["reason"] == "own_bank":
                        tagged_deposits[e["step"], e["holder_uid"]] += amount
                    dates.append([e["step"], e["reason"], e["holder_uid"], e["recipient_uid"], amount])
        same(sum(new.values()), q)
        routed[s]["physical_enemy_to_collector"] += q
        routed[s]["stolen_then_normal_own_deposit"] += new["own_bank"]
        routed[s]["stolen_still_carried_at_terminal"] += new["held_at_terminal"]
        routed[s]["enemy_worker_capture_lots"] += 1
        item["final_allocation"] = clean(new)
        item["downstream"] = dates
    for key, v in tagged_deposits.items():
        require(v <= observed_payments[key] + 1e-6, "stolen origins exceed actual same-worker deposit")
    term = certificate["official_reward_by_seat"]
    econ = []
    for s in (0, 1):
        same(start[s] + sum(total[s].values()), sim.cash[s])
        book = total[s].copy()
        book["official_terminal_adjustment"] = term[s] - sim.cash[s]
        econ.append(
            dict(
                cash=clean(book),
                tree_income=book["tree_absorption"] + book["tree_collision_cash"],
                enemy_carried_tax_in_tree_income=tree_tax[s],
                normal_deposit=book["own_delivery"],
                cargo=clean(routed[s]),
            )
        )
    score = round(term[seat] - term[1 - seat], 2)
    same(score, g["margin"])
    turns = deltas(cash_lead)
    obj = {
        "seed": seed,
        "seat": seat,
        "arm": arm,
        "map": certificate["map_sha256"],
        "raw": str(raw.relative_to(ROOT)),
        "raw_sha256": sha(raw),
        "pack_sha256": pack,
        "start_cash": start,
        "final_cash": list(sim.cash),
        "official_scores": term,
        "official_margin": score,
        "result": g["result"],
        "extinct_by_physical_seat": g["extinct_by_seat"],
        "turns": n,
        "cash_lead": cash_lead,
        "accounts_physical": econ,
        "capture_lots": received,
        "cash_turns": turns,
        "_frames": frames,
    }
    for h, peaks in turns.items():
        for k, x in peaks.items():
            x["own_minus_enemy_by_account"] = split_delta({arm: obj}, x["start"], x["end"])[arm]
    return obj


def bind(src):
    binding = (
        read(src / "games/binding.json")
        if (src / "games/binding.json").is_file()
        else read(src / "binding.json")
    )
    plan = read(src / "protocol.json")
    seeds = plan["new_dev"]["seeds"]
    if "parent" in binding and "candidate" in binding:
        return seeds, {a: binding[a] for a in ROLES}
    return seeds, {a: binding[a + "_pack_sha256"] for a in ROLES}


def paired(pair):
    a, b = pair["candidate"], pair["parent"]
    s = a["seat"]
    require(s == b["seat"] and a["map"] == b["map"], "pair mismatch")
    common = min(a["turns"], b["turns"])
    curve = [a["cash_lead"][i] - b["cash_lead"][i] for i in range(common + 1)]
    win = deltas(curve)
    selections = {arm: set() for arm in ROLES}

    def choose(arm, item):
        start, end = item["start"], item["end"]
        selections[arm].update(range(max(start - 10, 0), end))

    for h, extrema in win.items():
        for label, x in extrema.items():
            account = split_delta(pair, x["start"], x["end"])
            compare = Counter(account["candidate"])
            add_money(compare, account["parent"], -1)
            same(sum(compare.values()), x["change"])
            x["paired_cash_by_account"] = clean(compare)
            for arm in ROLES:
                choose(arm, x)
    for arm, g in pair.items():
        for peaks in g["cash_turns"].values():
            for x in peaks.values():
                choose(arm, x)
        # Preserve the observed approaches to the heaviest successful interception on either side.
        for side in (0, 1):
            stolen = [lot for lot in g["capture_lots"] if lot["acquirer_seat"] == side]
            if stolen:
                largest = max(stolen, key=lambda lot: lot["cargo"])
                t = largest["action_t"]
                choose(arm, {"start": t, "end": t + 1})
                largest["max_interception_for_this_seat"] = True
        g["turning_window_frames"] = [g["_frames"][i] for i in sorted(selections[arm])]
        del g["_frames"]
    delta = {}
    for who, idx in (("our", s), ("opponent", 1 - s)):
        diff = Counter(a["accounts_physical"][idx]["cash"])
        add_money(diff, b["accounts_physical"][idx]["cash"], -1)
        delta[who] = clean(diff)
    net = a["official_margin"] - b["official_margin"]
    same(sum(delta["our"].values()) - sum(delta["opponent"].values()), net)
    return dict(
        seed=a["seed"],
        seat=s,
        common_turns=common,
        terminal_margin_delta=net,
        accounts_differences=delta,
        delta_curve_cash_only=curve,
        paired_cash_turns=win,
        arms=pair,
    )


def audit(src):
    seeds, packs = bind(src)
    com = read(src / "comparison.json")
    finance = read(src / "income.json")
    facts = finance if isinstance(finance, list) else finance["games"]
    require(com["income_sha256"] == sha(src / "income.json") and len(facts) == 8, "income binding")
    pairs = []
    for seed in seeds:
        for seat in (0, 1):
            games = {}
            for a in ROLES:
                g = replay(src / "games" / a, seed, seat, a, packs[a])
                (i,) = [v for v in facts if (v["seed"], v["seat"], v["arm"]) == (seed, seat, a)]
                require(
                    i["raw_sha256"] == g["raw_sha256"]
                    and i["official_score_by_seat"] == g["official_scores"],
                    "different income transcript",
                )
                (matched,) = [v for v in com["pairs"] if (v["seed"], v["seat"]) == (seed, seat)]
                r = matched["arms"][a]
                require(
                    r["raw_sha256"] == g["raw_sha256"] and r["result"] == g["result"],
                    "comparison row mismatch",
                )
                same(r.get("margin", r.get("official_margin")), g["official_margin"])
                for s, ledger in enumerate(i["by_seat"]):
                    for key, z in ledger.items():
                        same(g["accounts_physical"][s]["cash"].get(key, 0), z)
                games[a] = g
            pairs.append(paired(games))
    maps = [mean(p["terminal_margin_delta"] for p in pairs if p["seed"] == seed) for seed in seeds]
    registered = com.get("paired_delta", com.get("paired_mean_margin_difference"))
    same(mean(maps), registered)
    opening = {}
    if src == HERE:
        sources = ("opening_teacher.json.gz", "opening_transfer.json.gz", "opening_coverage.json")
        teacher, transfer, coverage = (read(src / name) for name in sources)
        require(teacher["panel_sha256"] == transfer["panel_sha256"], "different opening archives")
        all_games = sum(row["maps"] for row in teacher["counts"].values())
        correct = sum(
            row["actual_by_physical_base_id"].get("RECCOLLECTOR/RECPLANTER", 0)
            for row in teacher["counts"].values()
        )
        cross = transfer["result"]["all"]
        n = cross["maps"]
        require(
            correct == all_games
            and cross["matched_full_teacher_command"] == {"candidate": 0, "parent": n, "control": n}
            and transfer["packages"]["candidate"] == packs["candidate"]
            and transfer["packages"]["parent"] == packs["parent"],
            "separate physical-base opening claims do not match binding",
        )
        opening = dict(
            archive_sha256={name: sha(src / name) for name in sources},
            all_maps=all_games,
            held=n,
            train_maps=teacher["counts"]["train"]["maps"],
            validation_maps=teacher["counts"]["val"]["maps"],
            holdout_maps=teacher["counts"]["holdout"]["maps"],
            trained_roots=coverage["actual_trained_roots"],
            trained_openings=coverage["actual_root0_count"],
            held_command_counts=cross,
        )
    return {
        "opening_context": opening,
        "protocol_sha256": sha(src / "protocol.json"),
        "comparison_sha256": sha(src / "comparison.json"),
        "income_sha256": sha(src / "income.json"),
        "source_sha256": sha(Path(__file__)),
        "lot_tracker_sha256": sha(BASE / "master6_actor_dev/lot_settlement_dev/lots.py"),
        "scope": "Historical signed games only. Both physical treasuries reconcile to official scores per game. A raid is counted only on actual rival-worker to collector transfer, even if immediately donated at the enemy base. For its current ownership spell, cargo accounting prorates tree tax, follows friendly relays and traces normal own-bank receipts separately. Reacquisition starts a new transfer. Own-bank cash can include these already captured tons and must never be added twice. Differences are map-seat matched independent games with uncontrolled genuine opponent responses and clock.",
        "map_deltas": maps,
        "mean_margin_delta": mean(maps),
        "pairs": pairs,
    }


ACCOUNT_NAMES = {
    "own_delivery": "正常回库",
    "tree_absorption": "树吸收",
    "tree_collision_cash": "树格碰撞到账",
    "hire_collector": "招采集工",
    "hire_planter": "招种植工",
    "plant": "种树付款",
    "seize": "夺树付款",
    "base_collision_cash": "基地碰撞到账",
    "opponent_worker_delivery_to_this_base": "对方入我基地交货",
    "rounding": "规则取整",
    "official_terminal_adjustment": "官方终局结算",
}


def by_policy(item, arm):
    return [p["arms"][arm] for p in item["pairs"]]


def detail(game, index):
    a = game["accounts_physical"][index]
    book = Counter(a["cash"])
    cargo = Counter(a["cargo"])
    return (
        a["normal_deposit"],
        a["tree_income"],
        a["enemy_carried_tax_in_tree_income"],
        -book["hire_collector"] - book["hire_planter"],
        -book["plant"],
        -book["seize"],
        cargo["physical_enemy_to_collector"],
        cargo["stolen_then_normal_own_deposit"],
    )


def describe_move(g, when, our_seat, target):
    """Read the real command and cell from each prior state for the UID credited in the swing."""
    timeline = {f["t"]: f for f in g["turning_window_frames"]}
    out = []
    for h in (10, 5, 3):
        t = when - h
        if t < 0:
            continue
        f = timeline[t]
        cell = f["pre_players"][our_seat][2].get(target)
        if cell:
            out.append(f"t{t} 格{cell[0]}、货{cell[1]:.0f}、{f['joint'][our_seat].get(target,'NONE')}")
        else:
            out.append(f"t{t} 未在场")
    return "，".join(out)


def report_markdown(item):
    games = {a: by_policy(item, a) for a in ROLES}
    names = {"candidate": "候选", "parent": "父代"}
    docs = ["# 双网融合候选对原版 Master3s2 的开发场经济复盘", ""]
    for arm in ("candidate", "parent"):
        gs = games[arm]
        w = Counter(g["result"] for g in gs)
        docs.append(
            f"{names[arm]}四场 {w['win']} 胜、{w['draw']} 平、{w['loss']} 负，平均官方终局分差 {mean(g['official_margin'] for g in gs):+.2f}。"
        )
    sem = stdev(item["map_deltas"]) / math.sqrt(len(item["map_deltas"]))
    docs.extend(
        [
            "",
            f"同图同席配对，候选平均终局分差较父代变化 **{item['mean_margin_delta']:+.2f}**。先对同一张图两席取平均，两张图分别为 {item['map_deltas'][0]:+.2f} 和 {item['map_deltas'][1]:+.2f}，按图聚类标准误 {sem:.2f}。仅有两个独立地图簇，正式独立考试记录为零场。",
            "",
            "| 地图种子 | Sapling 席位 | 父代官方分差 | 候选官方分差 | 候选减父代 | 父代/候选步数 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for p in item["pairs"]:
        parent = p["arms"]["parent"]
        candidate = p["arms"]["candidate"]
        docs.append(
            f"| {p['seed']} | {p['seat']} | {parent['official_margin']:+.2f} | {candidate['official_margin']:+.2f} | {p['terminal_margin_delta']:+.2f} | {parent['turns']}/{candidate['turns']} |"
        )
    o = item.get("opening_context")
    if o:
        n = o["held"]
        docs.extend(
            [
                "",
                "## 开局第一拍学到了什么",
                "",
                f"Master6 在 {o['train_maps']} 张训练图及另外 {n} 张地图（验证 {o['validation_maps']}、留出 {o['holdout_maps']}）的真实首拍，共 {o['all_maps']}/{o['all_maps']} 次在己方物理基地编号 0 招采集工、编号 1 招种植工。父代与同预算单网训练对照在这 {n} 张图全部逐基地命中。新候选在 {n}/{n} 张图均从两个基地各招一名采集工，首拍工种组合和编号 1 的指令都与示范有别。训练实际取到的 {o['trained_roots']:,} 个根中，首拍只占 {o['trained_openings']} 个。"
                + "这项赛后复查说明训练覆盖与首拍出令有落差，四场终局的胜负还受后续双方决策影响。[示范首拍](opening_teacher.json.gz) · [三包公开出令](opening_transfer.json.gz) · [实际训练抽样](opening_coverage.json)",
            ]
        )
    docs.extend(
        [
            "",
            "## 钱从哪里来",
            "",
            "每行取该版本四场的每场平均。树所得合并每步树吸收与树格碰撞到账，敌货树税已包含在树所得中。招工、种树和夺树列为实际支出。",
            "“截得货”只计真实敌工碰撞令货物进入一名本方采集工。“其中回库”跟随同一份本金，扣除沿路敌树抽取，允许友工接力，直至首次再次归敌、落地、其他结算、终局在途或正常回库。后者已包含在正常回库中。",
            "",
            "| 版本与收款方 | 正常回库 | 树所得 | 其中敌货树税 | 招工支出 | 种树支出 | 夺树支出 | 截得货 | 其中回库 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for a in ("parent", "candidate"):
        for side in (0, 1):
            vals = [detail(g, g["seat"] ^ side) for g in games[a]]
            avgs = [mean(col) for col in zip(*vals)]
            docs.append(
                "| "
                + names[a]
                + ("本方" if side == 0 else "场的考官")
                + " | "
                + " | ".join(f"{v:,.2f}" for v in avgs)
                + " |"
            )
    net = Counter()
    for p in item["pairs"]:
        add_money(net, p["accounts_differences"]["our"])
        add_money(net, p["accounts_differences"]["opponent"], -1)
    strongest = sorted(net, key=lambda k: -abs(net[k]))[:3]
    docs.extend(
        [
            "",
            "双方现金同时入账后，平均每场对终差变化贡献绝对值最大的三项为"
            + "、".join(ACCOUNT_NAMES.get(k, k) + f' {net[k]/len(item["pairs"]):+.2f}' for k in strongest)
            + "。其余项目和灭绝结算均在逐局账本内。",
            "",
            "## 同图同席的现金转折与此前的行动",
            "",
            "表内为每组两场仍在进行时，候选相对父代现金领先变化最快的五步。两个对手都实际重新应对，每次动作的终局因果贡献不能由两条已分叉轨迹单独识别。",
            "",
            "| 地图 / 席 | 五步最大上升 | 五步最大回吐 | 回吐窗贡献最大的账目 |",
            "|---|---:|---:|---|",
        ]
    )
    for p in item["pairs"]:
        peak = p["paired_cash_turns"]["5"]
        pos = peak.get("gain")
        neg = peak.get("loss")
        pretty = lambda x: f"{x['start']}–{x['end']}（{x['change']:+.0f}）" if x else "无"
        negterms = neg["paired_cash_by_account"] if neg else {}
        nk = min(negterms, key=negterms.get) if negterms else None
        docs.append(
            f"| {p['seed']} / {p['seat']} | {pretty(pos)} | {pretty(neg)} | "
            + (f"{ACCOUNT_NAMES.get(nk,nk)} {negterms[nk]:+.0f}" if nk else "无")
            + " |"
        )
    # Pick the single strongest cash swing per game and identify real worker deposits in its trace.
    docs.extend(
        [
            "",
            "以下按候选各局现金领先变化绝对值最大的五步，取该窗最大的一笔正常回库，向前还原交货工的指令、格号和携货。格号与方向沿用回放。",
            "",
        ]
    )
    for g in games["candidate"]:
        pool = g["cash_turns"].get("5", {})
        spike = max(pool.values(), key=lambda x: abs(x["change"]), default=None)
        if not spike:
            continue
        frames = {f["t"]: f for f in g["turning_window_frames"]}
        begin, end = spike["start"], spike["end"]
        candidates = [
            (e[2], t, e)
            for t in range(begin, end)
            for e in frames[t]["cash_events"]
            if e[0] == "own_delivery"
        ]
        predecessor = f"- {g['seed']} 席 {g['seat']}，t{begin}–{end} 现金领先变化 {spike['change']:+.2f}。"
        if candidates:
            # Look back from first action of the window, even for deliveries late in that window.
            amount, t, ev = max(candidates, key=lambda row: row[0])
            who = "本方" if ev[1] == g["seat"] else "考官"
            predecessor += (
                f"{who} {ev[3]} 在 t{t} 的行动后回库 {amount:.2f}。"
                + describe_move(g, t, ev[1], ev[3])
                + "。"
            )
        docs.append(predecessor)
    docs.extend(
        [
            "",
            "## 截货以后有多少兑现",
            "",
            "各候选局分别选本方和考官实抢货量最大的一笔。没有截货的方向记为无。括号中给出该批本金后来以本方正常回库到账的数量。",
            "",
            "| 地图 / 席 | 候选实抢（其中回库） | 考官实抢（其中回库） |",
            "|---|---|---|",
        ]
    )
    for g in games["candidate"]:
        cells = []
        for side in (g["seat"], 1 - g["seat"]):
            raids = [
                q
                for q in g["capture_lots"]
                if q["acquirer_seat"] == side and q.get("max_interception_for_this_seat")
            ]
            if not raids:
                cells.append("无")
                continue
            (q,) = raids
            cells.append(
                f"t{q['action_t']} 格{q['cell']}，{q['cargo']:.1f}（{q['final_allocation'].get('own_bank',0):.1f}）"
            )
        docs.append(f"| {g['seed']} / {g['seat']} | " + " | ".join(cells) + " |")
    docs.extend(
        [
            "",
            "[逐局现金、运货与双方行动证据](scorecard.json.gz)记录每局 1、3、5、10 步的最大现金上升与下降，及配对轨迹的对应极值。每个窗口保留最早前十步至窗口结束的双方联合命令、当期双方可见的碳格与数值、现金、工人及树状态。收据逐步与官方解释器交叉核对，双方现金账均闭合到终局分数。逐局首个分叉另有同根换令一拍的官方复演，以及双方开局出工、树息和回款记录。[首次分叉与开局账](first_split.json.gz) · [原始计分](comparison.json) · [现金流水](income.json)",
        ]
    )
    return "\n".join(docs) + "\n"


def save_report(raw):
    p = HERE / "RESULTS.md"
    if p.exists():
        require(p.read_bytes() == raw, "existing report differs")
        return
    tmp = p.with_suffix(".md." + str(os.getpid()) + ".part")
    tmp.write_bytes(raw)
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-check", action="store_true")
    args = ap.parse_args()
    src = ACTOR / "actor_partner_dev/replication" if args.prior_check else HERE
    item = audit(src)
    report = report_markdown(item).encode()
    if args.prior_check:
        print(len(item["pairs"]), item["mean_margin_delta"], len(report))
        return
    save(HERE / "scorecard.json.gz", item)
    save_report(report)


if __name__ == "__main__":
    main()
