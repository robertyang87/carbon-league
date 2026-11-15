#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S1 续弈验证 v3：SI07 落地（记忆恢复口径，取代 verify2/addendum 的冷启动口径）。

旧两版的病灶（REVIEW_SEARCH_IMPLEMENTATION SI07）：
  - 续弈臂用 fresh agent 冷启动：Feat 信念/历史、BankruptcyGuard 状
    态、对手模型记忆全部重置——两臂同等待遇只证明不了被测的是持续
    运行的生产续弈策略；
  - 终止沿用 step300（SI01 权威终局为终态 step299、最后有效动作观
    察 step298）。

本趟协议（采集/decide/续弈三段解耦）：
  1. 采集：基线自对弈（不挂 decide），诊断种子 986400 段。step≥240
     （硬线）且 key_units 触发的步，存根记录：sim 状态（get_state）、
     双席 obs、双席基线动作与步耗、本步前现金，以及**双席完整策略
     记忆的双份快照**——pre-call（该步 agent 调用前）与 post-call
     （调用后）。记忆构成（实测解剖，见 results 的
     memory_snapshot_composition）：
       · st 字典：f/fo 两个 Feat 信念体（est/last/last_step/hist/
         prev_w/prev_trees/inf/own_uid/aux）走厂商 feat_copy（其自用
         分支拷贝口径），并对 feat_copy 浅引用的 prev_w/prev_trees/
         aux 补 deepcopy 防跨臂污染；其余键（x/lg0/ot/oprev/rec/
         sampled/T_now/oenc/opp_first/guard_base 等，含对手模型状
         态）逐键 deepcopy，不可深拷贝者退化引用并登记键名；
       · BankruptcyGuard 实例状态（lostw/prevw/guard_base）deepcopy；
       · opp_model/opp_id 是嵌套函数（代码非状态），其状态全在
         st/fo 内，已随上两项覆盖；CachePolicy 的 last_enc 缓存是
         内容签名寻址（enc_for 签名不合即重编码），fresh 实例空缓
         存结果中性，不属记忆（由恢复复现检兜底验证）；
       · stats 字典仅遥测、不影响动作，不恢复（登记说明）。
     post-call 快照供离线 decide（decide 读的 x/lg0/f/fo 是调用后状
     态，与冻结版 decide_at_root 的在线口径一致）；pre-call 快照供
     续弈恢复（恢复后 agent 首次调用须从根观测开始处理，与生产形
     态一致）。
  2. 离线 decide：冻结版 harness.decide_at_root（SI01 终局拒绝＋
     SI02 唯一 deadline 口径，不用 s1_battery 的旧 decide_
     instrumented——它带已废的 0.05s 地板）。harness 的 inx 换成
     post-call 快照（st 为快照副本、guard 为根时刻守卫副本）。
     changed 根进验证，全局 ≥260 步优先、同档步数降序，至多 3 根。
  3. 续弈双臂：每臂一对 fresh agent，逐席 restore pre-call 记忆
     （st 内容原地换入 live 字典、guard __dict__ 换入快照状态），
     sim set_state 根状态。根步先调用双席恢复 agent 处理根观测
     （记忆按生产形态推进），根席执行臂动作（A0 臂＝基线动作／
     改子臂＝比较器所选动作）、对手席执行其调用返回动作；其后双
     席基线策略续弈至权威终局：step_count 到达 299（F04：真实终
     态现金）或灭绝（F01：终局步冻结、按带入灭绝步前现金）。守
     卫全程在位——续弈走 agent 本体 _inner（含 BankruptcyGuard
     逐歩 apply），非 real_env rollout 的近似。
     **恢复正确性硬检**：根步调用中，根席返回动作须与采集时该席
     基线动作（A0）逐字相等、对手席返回须与其记录动作相等；任一
     不等 → 该根记 restore_failed、差值无效（记忆快照不完整，不
     退回冷启动凑数）。
     两臂都到终局（299 或灭绝）才算有效差值；墙钟中断的臂记
     incomplete、差值 null。
  4. 镜像退化核查（必交行）：任一完整臂终局 cash[0]==cash[1] 精
     确相等、或两臂终局现金向量精确相等 → 比对两臂续弈动作序列
     逐步相同率与平局臂内双席席位归一化镜像率并给结论；未见精
     确相等明写「未复现」。
  5. 墙钟：全趟 ~1440 s 上限（本 VM 共享 CPU 参考值，非部署口
     径）；采集 ≤760 s、验证臂不晚于 ~1230 s 起跑、臂内截止
     ~1400 s。step≥240 为硬线，不降门槛凑数；收不满 3 个有效值
     如实报缺口。

运行：python3 s1_verify3.py
产出：s1_verify3_results.json。退出码 0 = 脚本跑完并落盘（单根成
败看 JSON/回执，不以退出码冒充验收结论）。
"""

import copy
import dataclasses
import hashlib
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

SEED0 = 986400
OUT_JSON = os.path.join(_HERE, "s1_verify3_results.json")
TERMINAL_STEP = real_env.TERMINAL_STEP     # 299（SI01 权威终局）

T_START = time.time()
HARD_WALL_S = 1440.0
SCAN_DEADLINE_S = 760.0
GAME2_MAX_START_WALL_S = 560.0
VERIFY_START_DEADLINE_S = 1230.0
ARM_DEADLINE_S = 1400.0

SNAP_MIN_STEP = 240                        # 硬线，不降
PREFER_STEP = 260
RECORD_CAP = 120
VERIFY_MAX_ROOTS = 3
EARLY_STOP_RECORDS = 10
EARLY_STOP_STEP = 268

_PACK_MOD = None


def wall():
    return time.time() - T_START


def make_fresh_agent(pack_path):
    """fresh 基线 agent（与 real_env.load_pack 构造逐字同口径：同包
    模块、每次重新反量化权重、make_search_agent 全新状态）。"""
    global _PACK_MOD
    import base64
    import zlib
    if _PACK_MOD is None:
        pack, _agent = real_env.load_pack(pack_path)
        _PACK_MOD = pack
    pack = _PACK_MOD
    w1 = pack._dequantize(zlib.decompress(base64.b64decode(pack._B1)), pack._Q1)
    w1["meta"] = np.array(pack._M1)
    return pack._S.make_search_agent(pack.CONF, weights=w1,
                                     rweights=None, oweights=None)


def _cells(fn):
    if fn.__closure__ is None:
        return {}
    return {name: cell.cell_contents
            for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)}


_FEAT_DEEP_EXTRAS = ("prev_w", "prev_trees", "aux")


def _copy_feat(fv, feat_copy, shallow, tag):
    """feat_copy ＋ 对其浅引用字段补 deepcopy（防跨臂污染）。"""
    c = feat_copy(fv)
    for extra in _FEAT_DEEP_EXTRAS:
        if hasattr(c, extra):
            try:
                setattr(c, extra, copy.deepcopy(getattr(fv, extra)))
            except Exception:
                shallow.append(f"{tag}.{extra}")
    return c


def snapshot_memory(agent, feat_copy):
    """单席完整策略记忆快照：st（f/fo 走 feat_copy＋补深拷贝、其余
    键 deepcopy）＋ BankruptcyGuard 实例状态 deepcopy。"""
    inner = _cells(agent)["_inner"]
    ci = _cells(inner)
    st = ci["st"]
    new = {}
    shallow = []
    for k, v in st.items():
        if k in ("f", "fo"):
            continue
        try:
            new[k] = copy.deepcopy(v)
        except Exception:
            new[k] = v
            shallow.append(k)
    for fk in ("f", "fo"):
        fv = st.get(fk)
        if fv is not None:
            new[fk] = _copy_feat(fv, feat_copy, shallow, fk)
    guard_state = copy.deepcopy(vars(ci["guard"]))
    return {"st": new, "guard": guard_state,
            "st_keys": sorted(new.keys()), "shallow_keys": shallow}


def build_st_copy(mem, feat_copy):
    """从已存快照再造一份独立 st 副本（快照本体保持 pristine）。"""
    src = mem["st"]
    out = {}
    shallow = []
    for k, v in src.items():
        if k in ("f", "fo"):
            continue
        out[k] = copy.deepcopy(v)
    for fk in ("f", "fo"):
        fv = src.get(fk)
        if fv is not None:
            out[fk] = _copy_feat(fv, feat_copy, shallow, fk)
    return out


def restore_agent(agent, mem, feat_copy):
    """把 pre-call 记忆恢复进 fresh agent：live st 字典内容原地换
    入（外层与 _inner 若持不同字典对象则各自换入一份独立副本）、
    guard 实例 __dict__ 换入快照状态。pol 缓存不恢复（内容签名寻
    址、结果中性）；stats 不恢复（仅遥测）。"""
    ca = _cells(agent)
    inner = ca["_inner"]
    ci = _cells(inner)
    targets = [ci["st"]]
    if ca.get("st") is not None and ca["st"] is not ci["st"]:
        targets.append(ca["st"])
    for t in targets:
        fresh = build_st_copy(mem, feat_copy)
        t.clear()
        t.update(fresh)
    g = ci["guard"]
    vars(g).clear()
    vars(g).update(copy.deepcopy(mem["guard"]))


def offline_decide(harness, rec, feat_copy):
    """在 post-call 快照上离线跑冻结版 decide_at_root。"""
    seat = rec["seat"]
    mem = rec["post"][seat]
    live = harness.inx
    guard_obj = copy.deepcopy(live.guard)
    vars(guard_obj).clear()
    vars(guard_obj).update(copy.deepcopy(mem["guard"]))
    harness.inx = dataclasses.replace(
        live, st=build_st_copy(mem, feat_copy), guard=guard_obj)
    try:
        obs_copy = copy.deepcopy(rec["obs"][seat])
        harness.begin_step(obs_copy)
        return harness.decide_at_root(
            obs_copy, rec["a0"],
            baseline_step_cost_s=rec["base_cost"], horizon=1,
            scenario_seed=rec["scenario_seed"])
    finally:
        harness.inx = live


def seq_hash(seq):
    raw = json.dumps(seq, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _norm_seat1(actions):
    out = {}
    for k, v in (actions or {}).items():
        out[k.replace("player-1-", "player-0-", 1)] = v
    return out


def run_arm(pack_path, FastCarbon, terminal_fn, feat_copy, rec, my_action):
    """一臂续弈：双席 restore pre-call 记忆 → 根步调用双席 agent 处
    理根观测（根席执行臂动作、对手执行其返回动作）→ 基线续弈至权
    威终局。返回臂记录（complete/终局现金/序列/复现检结果）。"""
    seat = rec["seat"]
    arm = {"complete": False, "extinct": False}
    t0 = time.time()
    try:
        agents = [make_fresh_agent(pack_path), make_fresh_agent(pack_path)]
        for s in (0, 1):
            restore_agent(agents[s], rec["pre"][s], feat_copy)
        sim = FastCarbon()
        sim.set_state(copy.deepcopy(rec["sim_state"]))
        if terminal_fn(sim):
            arm["incomplete_reason"] = "根状态已灭绝（不应到达：decide 已拒绝）"
            return arm
        # 根步：双席调用（记忆按生产形态处理根观测）＋复现硬检
        # obs 传深拷贝，防 agent 内部万一回写 obs 污染另一臂的根观测
        ret = [agents[0](copy.deepcopy(rec["obs"][0])),
               agents[1](copy.deepcopy(rec["obs"][1]))]
        arm["root_call_reproduced"] = {
            "root_seat": bool(ret[seat] == rec["a0"]),
            "opp_seat": bool(ret[1 - seat] == rec["opp_a0"]),
        }
        exec_actions = [ret[0], ret[1]]
        exec_actions[seat] = my_action
        arm["root_exec_actions"] = exec_actions
        cash_before = [float(sim.cash[0]), float(sim.cash[1])]
        sim.step(exec_actions[0], exec_actions[1])
        seq = []
        final = None
        if terminal_fn(sim):
            final = cash_before          # F01：灭绝步冻结
            arm["extinct"] = True
        while final is None:
            if sim.step_count >= TERMINAL_STEP:
                final = [float(sim.cash[0]), float(sim.cash[1])]   # F04
                break
            if wall() > ARM_DEADLINE_S:
                arm["incomplete_reason"] = (
                    f"验证墙钟截止（{ARM_DEADLINE_S:.0f}s）于 step "
                    f"{sim.step_count}")
                break
            cb = [float(sim.cash[0]), float(sim.cash[1])]
            b0 = agents[0](sim.obs(0))
            b1 = agents[1](sim.obs(1))
            seq.append([b0, b1])
            sim.step(b0, b1)
            if terminal_fn(sim):
                final = cb               # F01
                arm["extinct"] = True
        arm["seq"] = seq
        arm["seq_len"] = len(seq)
        arm["seq_hash"] = seq_hash(seq)
        arm["reached_step"] = sim.step_count
        if final is not None:
            arm["final_cash"] = final
            arm["final_diff_seat"] = float(final[seat] - final[1 - seat])
            arm["complete"] = True
        elif "incomplete_reason" not in arm:
            arm["incomplete_reason"] = f"只到 step {sim.step_count}"
    except Exception as exc:
        arm["incomplete_reason"] = f"异常：{exc!r}"
        arm.setdefault("reached_step", None)
    arm["wall_s"] = time.time() - t0
    return arm


def verify_root(pack_path, FastCarbon, terminal_fn, feat_copy, rec):
    """两臂续弈验证一根。恢复复现检不过 → restore_failed、差值无
    效；两臂都到终局才给有效差值。"""
    seat = rec["seat"]
    res = {"game": rec["game"], "seed": rec["seed"], "step": rec["step"],
           "seat": seat, "chosen_index": rec["dec"].chosen_index,
           "a0_action": rec["a0"], "chosen_action": rec["dec"].action,
           "opp_a0_action": rec["opp_a0"],
           "valid": False}
    arms = {}
    for name, act in (("a0", rec["a0"]), ("chosen", rec["dec"].action)):
        if wall() > VERIFY_START_DEADLINE_S and name == "a0":
            arms[name] = {"complete": False,
                          "incomplete_reason": "验证起跑墙钟已过，未跑"}
            continue
        arms[name] = run_arm(pack_path, FastCarbon, terminal_fn,
                             feat_copy, rec, act)
    res["arms"] = arms
    a, c = arms.get("a0", {}), arms.get("chosen", {})
    repro_ok = all(
        (arms[n].get("root_call_reproduced") or {}).get("root_seat")
        and (arms[n].get("root_call_reproduced") or {}).get("opp_seat")
        for n in ("a0", "chosen") if n in arms)
    res["restore_reproduction_ok"] = bool(repro_ok) and len(arms) == 2
    if not res["restore_reproduction_ok"]:
        res["invalid_reason"] = (
            "恢复复现检未过：恢复后根步调用未逐字复现采集时基线动作"
            "（记忆快照不完整），差值不成立、不退回冷启动")
        return res
    if a.get("complete") and c.get("complete"):
        res["chosen_minus_a0_final_diff"] = (c["final_diff_seat"]
                                             - a["final_diff_seat"])
        res["valid"] = True
    else:
        res["chosen_minus_a0_final_diff"] = None
        bad = [n for n, x in (("a0", a), ("chosen", c))
               if not x.get("complete")]
        res["invalid_reason"] = "臂未到终局：" + "、".join(
            f"{n}（{arms[n].get('incomplete_reason')}）" for n in bad)
    return res


def tie_analysis(verifications):
    """镜像退化核查（必交行）：精确相等触发 → 序列比对；未触发明
    写未复现。"""
    triggers = []
    for v in verifications:
        arms = v.get("arms") or {}
        for name, arm in arms.items():
            fc = arm.get("final_cash")
            if arm.get("complete") and fc and fc[0] == fc[1]:
                triggers.append({"kind": "within_arm_exact_tie",
                                 "root_step": v["step"], "seat": v["seat"],
                                 "arm": name, "final_cash": fc})
        a, c = arms.get("a0", {}), arms.get("chosen", {})
        if a.get("complete") and c.get("complete") \
                and a.get("final_cash") == c.get("final_cash"):
            triggers.append({"kind": "cross_arm_identical_final_cash",
                             "root_step": v["step"], "seat": v["seat"],
                             "final_cash": a.get("final_cash")})
    if not triggers:
        return {"seen": False,
                "conclusion": "未复现：本趟各完整臂终局现金无席内精确"
                              "平局、两臂终局现金向量亦无精确相等，"
                              "无镜像退化迹象。"}
    out = {"seen": True, "triggers": triggers, "details": []}
    seen_roots = set()
    for tr in triggers:
        key = (tr["root_step"], tr["seat"])
        if key in seen_roots:
            continue
        seen_roots.add(key)
        v = next(x for x in verifications
                 if x["step"] == tr["root_step"] and x["seat"] == tr["seat"])
        arms = v["arms"]
        a_seq = arms.get("a0", {}).get("seq") or []
        c_seq = arms.get("chosen", {}).get("seq") or []
        n = min(len(a_seq), len(c_seq))
        cross_ident = (sum(1 for i in range(n) if a_seq[i] == c_seq[i]) / n
                       if n else None)
        root_diff_uids = sum(
            1 for k in set(v["a0_action"]) | set(v["chosen_action"])
            if v["a0_action"].get(k) != v["chosen_action"].get(k))
        detail = {"root_step": v["step"], "seat": v["seat"],
                  "root_action_diff_uids": root_diff_uids,
                  "cross_arm_seq_identical_frac": cross_ident,
                  "continuation_steps_compared": n}
        tied_arm = tr.get("arm")
        if tied_arm and arms.get(tied_arm, {}).get("seq"):
            seq = arms[tied_arm]["seq"]
            mir = sum(1 for s0, s1 in seq
                      if _norm_seat1(s1) == (s0 or {})) / len(seq)
            detail["tied_arm"] = tied_arm
            detail["within_arm_mirror_frac"] = mir
        out["details"].append(detail)
    out["conclusion"] = (
        "见 details：cross_arm_seq_identical_frac 为两臂续弈逐步全同"
        "率（根步动作本有差异，若续弈全同且终局现金全等则两臂退化"
        "同轨）；within_arm_mirror_frac 为平局臂内双席动作席位归一"
        "化后逐步相同率（近 1 表示双席镜像/同构推进）。")
    return out


def collect_game(pack_path, game_idx, seed, feat_copy, terminal_fn,
                 key_units, cfg, FastCarbon):
    """基线自对弈一局（不挂 decide）；step≥240 触发步存根记录
    （pre/post 双份记忆快照）。返回 (records, info)。"""
    agents = [make_fresh_agent(pack_path), make_fresh_agent(pack_path)]
    sim = FastCarbon()
    sim.reset(seed=seed)
    records = []
    info = {"game": game_idx, "seed": seed, "records": 0,
            "record_steps": [], "stop_reason": None}
    while sim.step_count < TERMINAL_STEP:
        if wall() > SCAN_DEADLINE_S:
            info["stop_reason"] = f"采集墙钟截止（{SCAN_DEADLINE_S:.0f}s）"
            break
        step_now = sim.step_count
        obs = [sim.obs(0), sim.obs(1)]
        trig = [step_now >= SNAP_MIN_STEP and bool(key_units(obs[s], cfg))
                for s in (0, 1)]
        pre = post = sim_state = cash_before = None
        if any(trig) and not terminal_fn(sim):
            pre = [snapshot_memory(agents[0], feat_copy),
                   snapshot_memory(agents[1], feat_copy)]
            sim_state = sim.get_state()
            cash_before = [float(sim.cash[0]), float(sim.cash[1])]
        actions = [None, None]
        base_cost = [0.0, 0.0]
        for s in (0, 1):
            t0 = time.perf_counter()
            actions[s] = agents[s](obs[s])
            base_cost[s] = time.perf_counter() - t0
        if pre is not None:
            post = [snapshot_memory(agents[0], feat_copy),
                    snapshot_memory(agents[1], feat_copy)]
            for s in (0, 1):
                if trig[s]:
                    records.append({
                        "game": game_idx, "seed": seed, "seat": s,
                        "step": step_now,
                        "obs": copy.deepcopy(obs),
                        "sim_state": sim_state,
                        "cash_before": cash_before,
                        "pre": pre, "post": post,
                        "a0": copy.deepcopy(actions[s]),
                        "opp_a0": copy.deepcopy(actions[1 - s]),
                        "base_cost": base_cost[s],
                        "scenario_seed": seed + step_now,
                    })
            if len(records) > RECORD_CAP:
                records = records[-RECORD_CAP:]
            if len(records) >= EARLY_STOP_RECORDS and \
                    records[-1]["step"] >= EARLY_STOP_STEP:
                info["stop_reason"] = (
                    f"记录已足（{len(records)} 条、最高步 "
                    f"{records[-1]['step']}），提前收局")
                sim.step(actions[0], actions[1])
                break
        sim.step(actions[0], actions[1])
        if terminal_fn(sim):
            info["stop_reason"] = "灭绝终局（权威终局口径）"
            break
    info["reached_step"] = sim.step_count
    info["records"] = len(records)
    info["record_steps"] = sorted({r["step"] for r in records})
    return records, info


def main():
    print("== vendor md5 核验 ==")
    for k, v in real_env.verify_vendor().items():
        print(f"  {k}: {v}")
    pack_path = fetch_pack()
    _p, probe_agent = real_env.load_pack(pack_path)
    harness0 = real_env.RealEnvSearch(probe_agent, collector=StatsCollector())
    FastCarbon = harness0.inx.FastCarbon
    terminal_fn = harness0.inx.terminal_fn
    feat_copy = harness0.inx.mod["feat_copy"]
    key_units = harness0.inx.mod["key_units"]
    cfg = harness0.cfg

    # 记忆构成清单（回执必交）：实测解剖结果写死为自描述字段，
    # st 键全集在每条快照里另有 st_keys 实测登记。
    mem_composition = {
        "restored": [
            "st 字典全键（逐键 deepcopy；键全集见各快照 st_keys）："
            "含 f/fo 两个 Feat 信念体、x/lg0（根编码与先验）、"
            "ot/oprev/rec/sampled/T_now/oenc/opp_first（对手模型与"
            "教师状态）、guard_base/base_fix 等",
            "Feat f/fo：厂商 feat_copy（est/last/last_step/hist/"
            "own_uid/inf 私有拷贝）＋ prev_w/prev_trees/aux 补 "
            "deepcopy（feat_copy 对此三项是浅引用，补深防跨臂污染）",
            "BankruptcyGuard 实例状态：lostw/prevw/guard_base"
            "（__dict__ deepcopy 换入 fresh agent 的 live guard）",
        ],
        "not_restored_with_reason": [
            "CachePolicy.last_enc/last_sig：内容签名寻址缓存，签名"
            "覆盖 encode 全部输入字段、不合即重编码——fresh 空缓存"
            "只影响速度不影响输出，不属策略记忆（由复现检兜底）",
            "stats 字典：仅遥测计数，不影响动作",
            "_search 的 FastCarbon pool：rollout 草稿 sim，非记忆",
            "opp_model/opp_id/track_losses 等嵌套函数：是代码非状态，"
            "其状态（对手历史/风格识别）全在 st 与 fo 内、已恢复",
        ],
        "validation": "每臂根步调用复现检：恢复后 agent 对根观测的"
                      "返回动作须与采集时基线动作逐字相等（双席、"
                      "双臂各检一次）；不等 → 该根 restore_failed、"
                      "差值无效。",
    }

    doc = {"protocol": {
               "collect": "基线自对弈、不挂 decide；step≥240（硬线）"
                          "且 key_units 触发存根记录（sim 状态/双席 "
                          "obs/双席动作与步耗/pre-call 与 post-call "
                          "双份双席记忆快照）",
               "decide": "快照上离线跑冻结版 decide_at_root（SI01/"
                         "SI02 口径，post-call 快照换入 harness inx）",
               "verify": "双臂各一对 fresh agent、逐席 restore "
                         "pre-call 记忆；根步调用双席 agent 处理根观"
                         "测后根席执行臂动作；续弈至权威终局 "
                         "step299（F04）或灭绝冻结（F01）；守卫随 "
                         "agent 本体全程在位；根步复现检不过则该根"
                         "无效",
               "seeds": [SEED0, SEED0 + 1],
               "snap_min_step": SNAP_MIN_STEP,
               "prefer_step": PREFER_STEP,
               "verify_max_roots": VERIFY_MAX_ROOTS,
               "terminal_step": TERMINAL_STEP,
               "hard_wall_s": HARD_WALL_S},
           "memory_snapshot_composition": mem_composition,
           "scan_games": [], "changed_candidates": [],
           "selected_roots": [], "verifications": [], "tie_check": None,
           "notes": [
               "本趟口径与 verify2/addendum 的根本差别：续弈恢复双席"
               "完整策略记忆（非冷启动）、终止按 SI01 权威终局 "
               "step299（非 300）",
               "decide 用冻结版 decide_at_root，不用 s1_battery 的 "
               "decide_instrumented（旧 0.05s 地板口径，SI02 已废）",
               "scenario_seed 未消费（J=1 确定性），仅作登记标签",
               "叶①为原始分差尺度 affine 恒等直通（decide_at_root "
               "内口径）",
               "墙钟为本 VM 共享 CPU 参考值，不作部署口径",
           ]}

    harnesses = [real_env.RealEnvSearch(make_fresh_agent(pack_path),
                                        collector=StatsCollector()),
                 real_env.RealEnvSearch(make_fresh_agent(pack_path),
                                        collector=StatsCollector())]
    all_changed = []
    for g in range(2):
        seed = SEED0 + g
        if g == 1:
            if wall() > GAME2_MAX_START_WALL_S:
                doc["scan_games"].append(
                    {"game": 1, "seed": seed, "skipped": True,
                     "reason": f"墙钟 {wall():.0f}s 已过第二局起跑线"
                               f"（{GAME2_MAX_START_WALL_S:.0f}s）"})
                print(f"  局 1 跳过：墙钟 {wall():.0f}s")
                break
            if len(all_changed) >= VERIFY_MAX_ROOTS:
                doc["scan_games"].append(
                    {"game": 1, "seed": seed, "skipped": True,
                     "reason": f"第一局 changed 候选已足 "
                               f"（{len(all_changed)} 个），留墙钟给验证"})
                print("  局 1 跳过：changed 已足、留墙钟给验证")
                break
        print(f"== 采集局 {g}（seed {seed}）==")
        records, info = collect_game(pack_path, g, seed, feat_copy,
                                     terminal_fn, key_units, cfg,
                                     FastCarbon)
        n_decided = 0
        changed_here = []
        for rec in records:
            try:
                dec = offline_decide(harnesses[rec["seat"]], rec, feat_copy)
            except Exception as exc:
                rec["decide_error"] = repr(exc)
                continue
            if dec is None:
                rec["decide_none"] = True
                continue
            n_decided += 1
            rec["dec"] = dec
            if dec.changed:
                changed_here.append(rec)
        info["decided"] = n_decided
        info["changed"] = len(changed_here)
        all_changed.extend(changed_here)
        doc["scan_games"].append(info)
        print(f"  局 {g} 毕：到步 {info.get('reached_step')}、记录 "
              f"{info['records']}、离线 decide 成功 {n_decided}、"
              f"changed {len(changed_here)}、墙钟 {wall():.0f}s")

    all_changed.sort(key=lambda r: (0 if r["step"] >= PREFER_STEP else 1,
                                    -r["step"]))
    doc["changed_candidates"] = [
        {"game": r["game"], "seed": r["seed"], "step": r["step"],
         "seat": r["seat"], "chosen_index": r["dec"].chosen_index}
        for r in all_changed]
    selected = all_changed[:VERIFY_MAX_ROOTS]
    doc["selected_roots"] = [
        {"game": r["game"], "seed": r["seed"], "step": r["step"],
         "seat": r["seat"], "chosen_index": r["dec"].chosen_index,
         "a0_action": r["a0"], "chosen_action": r["dec"].action,
         "opp_a0_action": r["opp_a0"],
         "snapshot_st_keys": r["pre"][r["seat"]]["st_keys"],
         "snapshot_shallow_keys": r["pre"][r["seat"]]["shallow_keys"]}
        for r in selected]
    print(f"== 选根 {len(selected)} 个："
          f"{[(r['step'], r['seat']) for r in selected]} ==")

    verifications = []
    for rec in selected:
        print(f"== 续弈验证 seed={rec['seed']} seat={rec['seat']} "
              f"step={rec['step']} ==")
        try:
            verifications.append(verify_root(
                pack_path, FastCarbon, terminal_fn, feat_copy, rec))
        except Exception as exc:
            verifications.append(
                {"game": rec["game"], "seed": rec["seed"],
                 "step": rec["step"], "seat": rec["seat"],
                 "valid": False, "exception": repr(exc)})
    doc["verifications"] = verifications
    doc["tie_check"] = tie_analysis(verifications)
    doc["wall_s_total"] = wall()
    doc["valid_verifications"] = sum(1 for v in verifications
                                     if v.get("valid"))
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1, default=str)
    print(f"产出：{OUT_JSON}（总墙钟 {wall():.0f}s，有效验证 "
          f"{doc['valid_verifications']}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
