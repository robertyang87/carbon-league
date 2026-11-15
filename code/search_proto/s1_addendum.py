#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S1 电池补测：碰撞/逃逸精确正例（中局）+ 晚局 changed 根续弈验证。

承接 s1_battery 两处缺口：B 部分正例未达（P1/P2/P3 均 changed=False，
其中 P1/P2 是基线 A0 本就最优、P3 几何未产生分差）、verification 段
为空（电池 VERIFY_MIN_STEP=220 且 MAX_ROOTS=1，采集期内未存到根）。

一、碰撞/逃逸几何精确正例（注册战术型 1：即时货物碰撞/逃逸）：
   用 get_state/set_state 在真实 FastCarbon 构造中局状态（step
   100–250、left>0）。机制依据（vendor/fastsim.py Phase 2）：同格
   collector 中载货最少者胜、其余单位全灭且载货归胜者——满载
   collector 与轻载敌 collector 同格即丢货灭亡。几何族 A：敌轻载
   collector 站在我方满载 collector 归航路径格上（我方基地邻格、
   非基地格本身——基地格上 Phase 3 会把捕获货交付基地主，反成
   赠礼，电池 P2 已证）；几何族 D：敌轻载 collector 在我方满载
   collector 邻格、我方 collector 立于富碳格（A0 可能停采），由
   对手根动作扑入。每例先定型对手根动作 acts_o 再逐候选真跑一步
   双证：① comparator 的 leaf 分差；② 真实一步结局（key 单位
   存亡/载货/现金差、碰撞日志）。期望：A0 一步结局坏（key 单位灭
   或载货过半丢失）、比较器 changed=True 且选中逃逸候选（key 单位
   存活且载货 ≥90%）。多番构造不成则逐次落数据、定性回执，不硬凑。

二、晚局 changed 根续弈验证：自对弈深局扫到 step≥150 起收根，
   **step≥220 优先**（策略组判读口径），150–219 分档标注不混报。
   对 changed 根从根状态分两支（比较器所选 vs A0），其后双席均走
   基线策略（每臂一对 fresh agent，与电池 verify 同口径）续弈到
   300 步，比终局现金差（根座位口径），记所选支 − A0 支。

防失控：全脚本墙钟上限 ~590s（一 ≈170s、二扫描至 ~340s、验证至
~565s）；墙钟为本 VM 共享 CPU 参考值，不作部署口径。
种子：诊断块 986200–986499 内，扫描用 986360 起（电池已用
986200 段冒烟、986300 段电池 A/B）。

运行：python3 s1_addendum.py
产出：s1_addendum_results.json。退出码 0 = 脚本跑完（单件成败看
JSON/回执，不以退出码冒充）。
"""

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.expanduser("~/workspace"))

import real_env                            # noqa: E402
from telemetry import StatsCollector       # noqa: E402
from smoke_real_s1 import fetch_pack       # noqa: E402
from s1_battery import decide_instrumented  # noqa: E402

SEED_SCAN = 986360
OUT_JSON = os.path.join(_HERE, "s1_addendum_results.json")

T_START = time.time()
PART1_CAP_S = 170.0
SCAN_UNTIL_S = 340.0
VERIFY_UNTIL_S = 565.0
HARD_WALL_S = 590.0


def wall():
    return time.time() - T_START


# ================= 一、碰撞/逃逸正例 =================
def craft_mid(FastCarbon, step, cash, workers, carbon_map):
    """按 spec 造中局状态：无树，碳图按 carbon_map（其余 0）。"""
    sim = FastCarbon()
    sim.reset(seed=SEED_SCAN)
    snap = sim.get_state()
    snap["step"] = int(step)
    snap["cash"] = list(cash)
    carbon = [0.0] * 225
    for pos, v in (carbon_map or {}).items():
        carbon[int(pos)] = float(v)
    snap["carbon"] = carbon
    snap["trees"] = {}
    snap["trees_dict"] = {}
    snap["tid_counter"] = [0, 0]
    ws = {}
    wc = [0, 0]
    for wid, (p, pos, cargo, kind) in workers.items():
        ws[wid] = {"player": p, "pos": int(pos), "carbon": float(cargo),
                   "type": kind}
        suffix = int(str(wid).rsplit("-", 1)[1])
        wc[p] = max(wc[p], suffix + 1)
    snap["workers"] = ws
    snap["wid_counter"] = wc
    sim2 = FastCarbon()
    sim2.set_state(snap)
    assert sim2.step_count == int(step)
    return sim2, snap


def step_outcome(FastCarbon, snap, cand_action, acts_o, me, key_uid, e_uid):
    """真跑一步（真图），抽 key 单位结局与碰撞日志。"""
    s_ = FastCarbon()
    s_.set_state(snap)
    if me == 0:
        s_.step(cand_action, acts_o)
    else:
        s_.step(acts_o, cand_action)
    w = s_.workers.get(key_uid)
    ew = s_.workers.get(e_uid)
    coll = [{"wid": c[0], "player": c[1], "carbon": c[2], "gets": c[3],
             "winner_player": c[4]} for c in s_.last_collisions
            if c[0] in (key_uid, e_uid)]
    return {
        "key_alive": w is not None,
        "key_cargo_after": float(w.carbon) if w is not None else 0.0,
        "key_pos_after": int(w.pos) if w is not None else None,
        "enemy_alive": ew is not None,
        "enemy_cargo_after": float(ew.carbon) if ew is not None else 0.0,
        "cash": [float(s_.cash[0]), float(s_.cash[1])],
        "cash_diff_me": float(s_.cash[me] - s_.cash[1 - me]),
        "collisions_key": coll,
        "step_count": s_.step_count,
    }


def run_attempt(pack_path, FastCarbon, name, step, workers, carbon_map,
                me, key_uid, e_uid, cargo0, design_note):
    """一个构造例：fresh agent 冷启动 → A0 → 比较器 → 逐候选真结算。"""
    out = {"attempt": name, "design": design_note, "step": step, "me": me,
           "key_uid": key_uid, "enemy_uid": e_uid, "cargo0": cargo0}
    _pack, agent = real_env.load_pack(pack_path)
    harness = real_env.RealEnvSearch(agent, collector=StatsCollector())
    sim, snap = craft_mid(FastCarbon, step, [100, 100], workers, carbon_map)
    obs = sim.obs(me)
    harness.begin_step()
    t0 = time.perf_counter()
    a0 = agent(obs)
    base_cost = time.perf_counter() - t0
    out["baseline_step_cost_s"] = base_cost
    if not a0:
        out["error"] = "基线 agent 返回空动作（冷启动异常兜底）"
        return out
    out["a0_key_move"] = a0.get(key_uid, "STAY")
    dec, detail = decide_instrumented(harness, obs, a0, base_cost, horizon=1,
                                      scenario_seed=SEED_SCAN + 500)
    if dec is None:
        out["error"] = "无候选根（key_units 未触发或去重后候选 <2）"
        return out
    root = detail["root"]
    out["acts_o_enemy_key_move"] = root.acts_o.get(e_uid, "STAY")
    out["guard_vetoed"] = detail["guard_vetoed"]
    out["leaf_values"] = {str(k): v for k, v in detail["leaf_values"].items()}
    out["chosen_index"] = dec.chosen_index
    out["changed"] = bool(dec.changed)
    out["abort_reason"] = dec.abort_reason
    outcomes = {}
    for cand in detail["candidates"]:
        oc = step_outcome(FastCarbon, snap, cand.action, root.acts_o,
                          me, key_uid, e_uid)
        oc["key_move"] = cand.action.get(key_uid, "STAY")
        outcomes[cand.index] = oc
    out["outcomes"] = {str(k): v for k, v in outcomes.items()}
    a0oc = outcomes[0]
    choc = outcomes[dec.chosen_index]
    out["a0_bad"] = (not a0oc["key_alive"]) or \
        a0oc["key_cargo_after"] <= 0.5 * cargo0
    out["chosen_escape_ok"] = bool(choc["key_alive"]) and \
        choc["key_cargo_after"] >= 0.9 * cargo0
    leaf = detail["leaf_values"]
    if leaf and 0 in leaf and dec.chosen_index in leaf:
        out["leaf_chosen_minus_a0"] = leaf[dec.chosen_index] - leaf[0]
    out["success"] = bool(dec.changed) and out["a0_bad"] and \
        out["chosen_escape_ok"]
    return out


def part1(pack_path, FastCarbon, F):
    probe = FastCarbon()
    probe.reset(seed=SEED_SCAN)
    base0 = probe.bases[0][0][1]
    base0b = probe.bases[0][1][1]
    base1 = probe.bases[1][0][1]
    base1b = probe.bases[1][1][1]
    key_uid = "player-0-worker-0"
    e_uid = "player-1-worker-0"
    extras_me = {
        "player-0-worker-1": (0, base0b, 0, "COLLECTOR"),
        "player-0-worker-2": (0, base0b, 0, "COLLECTOR"),
    }
    extra_en = {"player-1-worker-1": (1, base1b, 0, "COLLECTOR")}

    attempts = []
    success = None

    def try_one(name, step, workers, carbon_map, cargo0, note):
        nonlocal success
        if success is not None or wall() > PART1_CAP_S:
            return
        try:
            r = run_attempt(pack_path, FastCarbon, name, step, workers,
                            carbon_map, 0, key_uid, e_uid, cargo0, note)
        except Exception as exc:
            r = {"attempt": name, "design": note, "exception": repr(exc)}
        attempts.append(r)
        print(f"  [{name}] changed={r.get('changed')} "
              f"a0_bad={r.get('a0_bad')} esc_ok={r.get('chosen_escape_ok')} "
              f"success={r.get('success')} err={r.get('error')} "
              f"exc={r.get('exception')}")
        if r.get("success"):
            success = r

    # 几何族 A：敌轻载站归航路径格（基地邻格、非基地格），我方满载在其外侧
    ys = [c for c in probe._nbr4[base0] if float(F.DIST[c, base0]) == 1.0]
    a_pairs = []
    for y in ys:
        for x in probe._nbr4[y]:
            if float(F.DIST[x, base0]) == 2.0 and x != base0:
                a_pairs.append((x, y))
    for cargo0, carbon_e, tag in ((40.0, 100.0, "A1"), (60.0, 100.0, "A2"),
                                  (40.0, 60.0, "A3")):
        for i, (x, y) in enumerate(a_pairs):
            workers = {
                key_uid: (0, x, cargo0, "COLLECTOR"),
                e_uid: (1, y, 2, "COLLECTOR"),
                **extras_me, **extra_en,
            }
            try_one(f"{tag}_x{x}_y{y}", 150, workers, {y: carbon_e}, cargo0,
                    f"族A：我方 collector({cargo0:g}) 在 {x}（距基地 {base0} 2 格），"
                    f"敌 collector(2) 在归航格 {y}（基地邻格）下碳 {carbon_e:g}；"
                    f"A0 若踏入 {y} → 碰撞少载者(敌2)胜、我方灭且货归敌；"
                    f"期望 changed=True 且选中绕行/停留逃逸候选。")
    # 几何族 D：我方满载立富碳格停采，敌轻载邻格扑入（靠对手根动作）
    for i, (x, y) in enumerate(a_pairs[:4]):
        for e_pos in probe._nbr4[x]:
            if e_pos in (y, base0):
                continue
            workers = {
                key_uid: (0, x, 40, "COLLECTOR"),
                e_uid: (1, e_pos, 2, "COLLECTOR"),
                **extras_me, **extra_en,
            }
            try_one(f"D_x{x}_e{e_pos}", 150, workers, {x: 100.0}, 40.0,
                    f"族D：我方 collector(40) 在富碳格 {x}（碳 100，A0 可能停采），"
                    f"敌 collector(2) 邻格 {e_pos}；若对手根动作扑入 {x} 且 A0 停留 "
                    f"→ 碰撞我方灭货归敌；期望比较器改选离开候选。")
    verdict = ("正例达成" if success else
               "正例未达：逐次尝试数据见 attempts，定性见回执")
    return {"attempts": attempts, "success": success, "verdict": verdict,
            "base0": base0, "a_pairs": a_pairs}


# ================= 二、晚局续弈验证 =================
def verify_root(pack_path, entry):
    """两支续弈（电池 verify 同口径：每臂一对 fresh 基线 agent）。"""
    res = {"game": entry["game"], "seat": entry["seat"],
           "step": entry["step"], "band": entry["band"]}
    FastCarbon = entry["FastCarbon"]
    arms = {}
    for arm_name, my_action in (("a0", entry["a0"]),
                                ("chosen", entry["chosen"])):
        if wall() > VERIFY_UNTIL_S:
            arms[arm_name] = {"error": "验证墙钟上限前未跑"}
            continue
        _p1, va0 = real_env.load_pack(pack_path)
        _p2, va1 = real_env.load_pack(pack_path)
        sim = FastCarbon()
        sim.set_state(entry["snap"])
        if entry["seat"] == 0:
            sim.step(my_action, entry["other"])
        else:
            sim.step(entry["other"], my_action)
        agents = (va0, va1)
        complete = True
        while sim.step_count < 300:
            if wall() > VERIFY_UNTIL_S:
                complete = False
                break
            b0 = agents[0](sim.obs(0))
            b1 = agents[1](sim.obs(1))
            sim.step(b0, b1)
        arms[arm_name] = {
            "final_cash": [float(sim.cash[0]), float(sim.cash[1])],
            "final_diff_seat": float(sim.cash[entry["seat"]]
                                     - sim.cash[1 - entry["seat"]]),
            "reached_step": sim.step_count,
            "complete": complete and sim.step_count >= 300,
        }
    res["arms"] = arms
    a, c = arms.get("a0", {}), arms.get("chosen", {})
    if a.get("complete") and c.get("complete"):
        res["chosen_minus_a0_final_diff"] = (c["final_diff_seat"]
                                             - a["final_diff_seat"])
    else:
        res["chosen_minus_a0_final_diff"] = None
        res["incomplete_reason"] = "至少一臂未续弈到 300 步（墙钟）"
    return res


def part2(pack_path):
    _p, agent0 = real_env.load_pack(pack_path)
    _p2, agent1 = real_env.load_pack(pack_path)
    agents = (agent0, agent1)
    collectors = (StatsCollector(), StatsCollector())
    harnesses = (real_env.RealEnvSearch(agent0, collector=collectors[0]),
                 real_env.RealEnvSearch(agent1, collector=collectors[1]))
    FastCarbon = harnesses[0].inx.FastCarbon

    scan = {"games": [], "roots_total": 0, "changed_total": 0,
            "changed_ge220": 0, "changed_150_219": 0}
    saved = []          # 待验证（内存含 snap，不入 JSON）
    saved_meta = []     # 入 JSON 的元数据
    games_done = 0
    for g in range(3):
        if wall() > SCAN_UNTIL_S:
            break
        if sum(1 for s in saved if s["band"] == "ge220") >= 4:
            break
        games_done += 1
        sim = FastCarbon()
        sim.reset(seed=SEED_SCAN + g)
        ginfo = {"game": g, "seed": SEED_SCAN + g, "roots": 0, "changed": 0}
        # 快进到 150 步（不收根；基线动作照常推进，agent 状态自然演化）
        while sim.step_count < 150 and wall() <= SCAN_UNTIL_S:
            a_0 = agents[0](sim.obs(0))
            a_1 = agents[1](sim.obs(1))
            sim.step(a_0, a_1)
        # 150 起双席收根
        while sim.step_count < 300 and wall() <= SCAN_UNTIL_S:
            step_now = sim.step_count
            actions = [None, None]
            decs = [None, None]
            for seat in (0, 1):
                obs = sim.obs(seat)
                harnesses[seat].begin_step()
                t0 = time.perf_counter()
                act = agents[seat](obs)
                base_cost = time.perf_counter() - t0
                actions[seat] = act
                dec, detail = decide_instrumented(
                    harnesses[seat], obs, act, base_cost, horizon=1,
                    scenario_seed=SEED_SCAN + 700 + step_now)
                if dec is not None:
                    decs[seat] = dec
                    ginfo["roots"] += 1
                    scan["roots_total"] += 1
                    if dec.changed:
                        ginfo["changed"] += 1
                        scan["changed_total"] += 1
            for seat in (0, 1):
                if decs[seat] is not None and decs[seat].changed:
                    band = "ge220" if step_now >= 220 else "150_219"
                    if band == "ge220":
                        scan["changed_ge220"] += 1
                    else:
                        scan["changed_150_219"] += 1
                    cap = 5 if band == "ge220" else 2
                    have = sum(1 for s in saved if s["band"] == band)
                    if have < cap:
                        saved.append({
                            "game": g, "seat": seat, "step": step_now,
                            "band": band, "snap": sim.get_state(),
                            "a0": actions[seat], "chosen": decs[seat].action,
                            "other": actions[1 - seat],
                            "FastCarbon": FastCarbon,
                        })
                        saved_meta.append({
                            "game": g, "seat": seat, "step": step_now,
                            "band": band,
                            "a0_action": actions[seat],
                            "chosen_action": decs[seat].action,
                        })
                        print(f"  收 changed 根 g{g} seat{seat} "
                              f"step={step_now} band={band}")
            sim.step(actions[0], actions[1])
        ginfo["reached_step"] = sim.step_count
        scan["games"].append(ginfo)
        print(f"  局 {g} 扫描毕：根 {ginfo['roots']}、changed {ginfo['changed']}、"
              f"到步 {sim.step_count}、墙钟 {wall():.0f}s")
        if sum(1 for s in saved if s["band"] == "ge220") >= 3 \
                and wall() > 200:
            break

    # 验证顺序：≥220 优先（步数降序，续弈便宜者先），再 150–219
    order = sorted([s for s in saved if s["band"] == "ge220"],
                   key=lambda s: -s["step"]) + \
        sorted([s for s in saved if s["band"] == "150_219"],
               key=lambda s: -s["step"])
    verifications = []
    for entry in order:
        if len(verifications) >= 3 or wall() > VERIFY_UNTIL_S:
            break
        print(f"== 续弈验证 g{entry['game']} seat{entry['seat']} "
              f"step={entry['step']} band={entry['band']} ==")
        try:
            verifications.append(verify_root(pack_path, entry))
        except Exception as exc:
            verifications.append({"game": entry["game"], "seat": entry["seat"],
                                  "step": entry["step"], "band": entry["band"],
                                  "exception": repr(exc)})
    scan["games_done"] = games_done
    return {"scan": scan, "saved_roots": saved_meta,
            "verifications": verifications}


def main():
    print("== vendor md5 核验 ==")
    for k, v in real_env.verify_vendor().items():
        print(f"  {k}: {v}")
    pack_path = fetch_pack()
    _p, _agent = real_env.load_pack(pack_path)
    harness0 = real_env.RealEnvSearch(_agent, collector=StatsCollector())
    FastCarbon = harness0.inx.FastCarbon
    F = harness0.inx.F

    print("== 一、碰撞/逃逸几何精确正例 ==")
    p1 = part1(pack_path, FastCarbon, F)
    print(f"  一 verdict：{p1['verdict']}（尝试 {len(p1['attempts'])} 次，"
          f"墙钟 {wall():.0f}s）")

    print("== 二、晚局 changed 根续弈验证 ==")
    p2 = part2(pack_path)
    print(f"  二 scan：{json.dumps(p2['scan'], ensure_ascii=False)}")
    for v in p2["verifications"]:
        print(f"  验证 g{v.get('game')} seat{v.get('seat')} step={v.get('step')} "
              f"band={v.get('band')} chosen-a0={v.get('chosen_minus_a0_final_diff')}")

    doc = {
        "part1_collision_escape": p1,
        "part2_late_verification": p2,
        "wall_s_total": wall(),
        "notes": [
            "对手根动作 acts_o 在每例根上定型后才逐候选真跑一步（双证口径同电池 B）",
            "叶①为原始分差尺度 affine 恒等直通；leaf_values 即原始分差",
            "续弈验证每臂一对 fresh 基线 agent（电池 verify 同口径，信念冷启动两臂同等待遇）",
            "step 分档：ge220 = step≥220（策略组口径优先档），150_219 = 150–219，两档不混报",
            "墙钟为本 VM 共享 CPU 参考值，不作部署口径",
        ],
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    print(f"产出：{OUT_JSON}（总墙钟 {wall():.0f}s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
