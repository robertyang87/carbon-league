#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 双臂运行器（冻结协议 §7/§9；B/C 确认件 13d3f3b4；快路径裁定
9651f30b §二：归因先导形态）。

逐根流程（根来自抽样补充件的入样根单，按局分片、按 root_id 断点续跑）：
  0. 采集：该局基线自对弈重放（基线包 819250bf，/tmp 预置优先），
     在目标步对双席取 pre/post-call 完整记忆快照＋sim 状态＋双席
     obs/动作（s1_verify3 collect 同形态，在线处理、快照只留当前步）。
  1. 触发复核：key_units 须仍含该根 key_uid 且 stake 逐字合
     （不合＝根被证伪 → invalid）。
  2. §6 准入核对（仅可辨根）：按枚举同口径复算合法候选表（生成序、
     守卫 veto 剔除）与先验截断准入集；冻结 settlement_argmax 不在
     准入集 → invalid（admission_floor_violation），不跑 decide/双臂。
  3. 叶差 margin（H=1 轮值复算）：在 post-call 快照上以冻结 harness
     机械（real_env._gen_candidates_raw/_guard_screen/_make_rollout
     ＋叶①恒等 adapter）复算该根比较器 H=1 轮的逐候选叶值——与内核
     _cmp_rollout/_raw_leaf 在 H=1 逐项同口径（根步一推进、终局种类
     分流、prev_cash 取入根步前现金）；margin ＝ 合格非 A0 候选叶值
     最大值 − A0 叶值（原始分差尺度）。无候选/无合格非 A0 时
     margin=null 并记 margin_status。
  4. A 臂决策（权威比较器）：resync 包（md5 cba5923c，内嵌五源逐件
     验合：comparator 385190bd／search c3222eea／feat 2a45b08b／
     np d8c15ad4／fastsim aefe6c83；CONF 生产形 SEARCH=True、
     CMP_H=1、BUDGET=0.7）的 fresh agent 恢复该根 pre-call 快照后
     对根 obs 调用一次——其返回值即 A 臂动作；cmp_stats 前后差分
     记遥测（calls/completed_rounds/changed/leaf_calls/abort）。
     内核 agent 逐局一只（cmp_chi 桶按局内步序累积，同 verify3
     harness 复用形态）；changed ＝ A 动作 ≠ A0（逐字比较）。
     harness 轮值 argmax 与内核动作逐根交叉记录（不一致且内核未
     abort 时记 unexplained_mismatch 供分析层点名）。
  5. 双臂续弈（verify3 形态）：双席 fresh 基线 agent 恢复 pre-call
     快照，根步双席调用＋复现硬检（双席返回须逐字复现采集动作，
     不过 → invalid restore_failed），根席执行臂动作（A0 臂＝A0、
     A 臂＝比较器动作），其后双席基线自博弈至权威终局（step 299
     F04 终态现金／灭绝 F01 权威逐席分数）。逐臂记双席终局分数对
     ＋终局类型。任一臂未到终局 → invalid（绝不记 0）。
  6. 落盘：逐根 append＋flush 到 results_<shard>.jsonl；重跑时
     已记录 root_id 跳过（--retry-invalid 可重跑 invalid 根）。

成本：逐根记 decide/margin/双臂/总墙钟；--pilot 时汇总 mean/p95/max
并按裁定 30 秒停线标注（本机 VM 口径不作考场读数）。

禁区：冻结根单/补充件只读；proto 冻结件只 import 不改；本脚本不算
ΔU（分析层 estimand.py 由分数对复算，§7.4）。
"""

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import sys
import time

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.expanduser("~/workspace/search-group/proto")
_S3F = os.path.expanduser("~/workspace/search-group/s3_freeze")
for _p in (_PROTO, _S3F, _HERE, os.path.expanduser("~/workspace")):
    sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import real_env      # noqa: E402

# s1_verify3 的导入链含 smoke_real_s1，其模块级按位置解析 sys.argv
# （ROOTS_TARGET/MAX_GAMES）——导入前暂存并清空位置参数、导入后恢复
# （枚举脚本 enumerate_roots 为此干脆不 import s1_verify3；本器需要
# 其快照/恢复机械，故走消毒导入）。
_saved_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    import s1_verify3    # noqa: E402
finally:
    sys.argv = _saved_argv

from enumerate_roots import Enumerator, fetch_pack  # noqa: E402

RESYNC_PACK_DEFAULT = os.path.expanduser(
    "~/workspace/search-group/resync/build/echo4_search_int8_resync.py")
RESYNC_PACK_MD5 = "cba5923c13fd629208a4991811723980"
RESYNC_SRC_MD5 = {
    "cnet2_feat": "2a45b08b6f97508541e62867e408748e",
    "cnet2_np": "d8c15ad4fdd8ee079363cf215d52daf3",
    "fastsim": "aefe6c83826efae4de83bef8c643c80b",
    "cnet2_comparator": "385190bd2f59d90f4e68a0d2dc6091a1",
    "cnet2_search": "c3222eea76ea9d42c160626c2e239ca0",
}
TERMINAL_STEP = 299
STAKE_TOL = 1e-9


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def load_resync_pack(path: str):
    """加载 resync 包并逐件验合（整包 md5＋内嵌五源 md5＋CONF 关键项）。"""
    import importlib.util
    with open(path, "rb") as fh:
        raw = fh.read()
    if _md5(raw) != RESYNC_PACK_MD5:
        raise RuntimeError(f"resync 包 md5 不合：{_md5(raw)} —— 停")
    spec = importlib.util.spec_from_file_location("s3_resync_pack", path)
    pack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pack)
    for name, want in RESYNC_SRC_MD5.items():
        got = _md5(pack._SRC[name].encode("utf-8"))
        if got != want:
            raise RuntimeError(f"resync 内嵌源 {name} md5 不合："
                               f"{got} vs {want} —— 停")
    conf = pack.CONF
    if not (conf.get("SEARCH") is True and conf.get("CMP_H") == 1
            and conf.get("BUDGET") == 0.7):
        raise RuntimeError(f"resync 包 CONF 非生产钉版形：{conf} —— 停")
    return pack


def make_kernel_agent(pack):
    """resync 包 fresh agent（与基线包同构造口径：_B1 反量化＋
    make_search_agent 全新状态）。"""
    import base64
    import zlib
    w = pack._dequantize(zlib.decompress(base64.b64decode(pack._B1)),
                         pack._Q1)
    w["meta"] = np.array(pack._M1)
    return pack._S.make_search_agent(pack.CONF, weights=w,
                                     rweights=None, oweights=None)


class Ctx:
    def __init__(self, pack_path, resync_path, arm_cap_s):
        self.pack_path = pack_path
        self.arm_cap_s = arm_cap_s
        _p, probe_agent = real_env.load_pack(pack_path)
        probe_harness = real_env.RealEnvSearch(probe_agent)
        self.inx = probe_harness.inx
        self.cfg = self.inx.cfg
        self.FastCarbon = self.inx.FastCarbon
        self.terminal_fn = self.inx.terminal_fn
        self.feat_copy = self.inx.mod["feat_copy"]
        self.key_units = self.inx.mod["key_units"]
        self.score_fn = s1_verify3.make_terminal_score_fn(self.inx)
        self.resync_pack = load_resync_pack(resync_path)


def prepare_harness(harness, rec, feat_copy):
    """把 harness 换到该根 post-call 快照（s1_verify3.offline_decide
    的 inx 换装同口径：st/guard 取根席 post-call、对手 guard 取对手
    post-call、_fo_snap 取根席 pre-call fo）。返回 cleanup()。"""
    seat = rec["seat"]
    mem = rec["post"][seat]
    opp_mem = rec["post"][1 - seat]
    live = harness.inx
    guard_obj = copy.deepcopy(live.guard)
    vars(guard_obj).clear()
    vars(guard_obj).update(copy.deepcopy(mem["guard"]))
    opp_guard_obj = copy.deepcopy(live.guard)
    vars(opp_guard_obj).clear()
    vars(opp_guard_obj).update(copy.deepcopy(opp_mem["guard"]))
    harness.inx = dataclasses.replace(
        live, st=s1_verify3.build_st_copy(mem, feat_copy),
        guard=guard_obj)
    prev_opp_guard = harness._opp_guard
    harness.set_opponent_guard(opp_guard_obj)
    prev_fo_snap = harness._fo_snap
    pre_st = (rec.get("pre") or [None, None])[seat]
    pre_fo = pre_st["st"].get("fo") if pre_st else None
    if pre_fo is not None:
        shallow = []
        fo_snap = s1_verify3._copy_feat(pre_fo, feat_copy, shallow, "fo")
        if int(getattr(fo_snap, "seat", -1)) != 1 - seat:
            raise RuntimeError("pre-call 快照 fo 席位不合 —— 停")
        harness._fo_snap = fo_snap
    else:
        harness.begin_step(copy.deepcopy(rec["obs"][seat]))

    def cleanup():
        harness.inx = live
        harness._opp_guard = prev_opp_guard
        harness._fo_snap = prev_fo_snap

    return cleanup


def run_arm_s3(ctx: Ctx, rec, my_action):
    """一臂续弈（s1_verify3.run_arm 同形态；墙钟改本臂本地上限，
    不借 verify3 的全局墙钟）。终局口径：299 到时＝F04 终态现金；
    灭绝＝F01 权威逐席分数（灭绝优先）。"""
    seat = rec["seat"]
    arm = {"complete": False, "extinct": False}
    t0 = time.monotonic()
    deadline = t0 + ctx.arm_cap_s
    try:
        agents = [s1_verify3.make_fresh_agent(ctx.pack_path),
                  s1_verify3.make_fresh_agent(ctx.pack_path)]
        for s in (0, 1):
            s1_verify3.restore_agent(agents[s], rec["pre"][s],
                                     ctx.feat_copy)
        sim = ctx.FastCarbon()
        sim.set_state(copy.deepcopy(rec["sim_state"]))
        if ctx.terminal_fn(sim):
            arm["incomplete_reason"] = "根状态已灭绝（不应到达）"
            return arm
        ret = [agents[0](copy.deepcopy(rec["obs"][0])),
               agents[1](copy.deepcopy(rec["obs"][1]))]
        arm["root_call_reproduced"] = {
            "root_seat": bool(ret[seat] == rec["a0"]),
            "opp_seat": bool(ret[1 - seat] == rec["opp_a0"]),
        }
        exec_actions = [ret[0], ret[1]]
        exec_actions[seat] = my_action
        cash_before = [float(sim.cash[0]), float(sim.cash[1])]
        sim.step(exec_actions[0], exec_actions[1])
        final = None
        final_basis = None
        if ctx.terminal_fn(sim):
            final = ctx.score_fn(sim, cash_before)
            final_basis = "F01_terminal_scores"
            arm["extinct"] = True
        while final is None:
            if sim.step_count >= TERMINAL_STEP:
                final = [float(sim.cash[0]), float(sim.cash[1])]
                final_basis = "F04_terminal_cash"
                break
            if time.monotonic() > deadline:
                arm["incomplete_reason"] = (
                    f"臂墙钟上限 {ctx.arm_cap_s:.0f}s 于 step "
                    f"{sim.step_count}")
                break
            cb = [float(sim.cash[0]), float(sim.cash[1])]
            b0 = agents[0](sim.obs(0))
            b1 = agents[1](sim.obs(1))
            sim.step(b0, b1)
            if ctx.terminal_fn(sim):
                final = ctx.score_fn(sim, cb)
                final_basis = "F01_terminal_scores"
                arm["extinct"] = True
        arm["reached_step"] = sim.step_count
        if final is not None:
            arm["final_scores"] = [float(final[0]), float(final[1])]
            arm["final_basis"] = final_basis
            arm["terminal_type"] = ("extinction" if arm["extinct"]
                                    else "time_limit")
            if arm["extinct"]:
                arm["f01_scores"] = arm["final_scores"]
            arm["complete"] = True
    except Exception as exc:
        arm["incomplete_reason"] = f"异常：{exc!r}"
        arm.setdefault("reached_step", None)
    arm["wall_s"] = time.monotonic() - t0
    return arm


def admission_legal_set(harness, root_ctx, a0):
    """§6 准入复算（枚举同口径）：全量候选（生成序）→守卫 veto 剔除
    得合法表→先验截断 keep（全量坐标）→准入集（合法坐标集合）。
    返回 (legal_count, admission_legal_set, keep_full_indices)。"""
    full = Enumerator._full_candidates(None, harness, root_ctx, a0)
    if full is None:
        return 1, {0}, [0]
    cand_actions, cand_labels, cand_meta = full
    frozen = harness._gen_candidates_raw(root_ctx, a0)
    if frozen is None:
        raise RuntimeError("复制管线有候选、冻结函数无候选 —— 实现漂移，停")
    uid2slot = {u: s for s, u in enumerate(root_ctx.f.own_uid)
                if u is not None}
    tr_actions, _tr_labels = Enumerator._truncate(
        cand_actions, cand_labels, cand_meta, root_ctx.lg0, uid2slot)
    f_actions, _f_labels = frozen
    if len(tr_actions) != len(f_actions) or any(
            tr_actions[i] != f_actions[i] for i in range(len(tr_actions))):
        raise RuntimeError("截断交叉核对不等 —— 实现漂移，停")
    id2idx = {id(a): i for i, a in enumerate(cand_actions)}
    keep = sorted(id2idx[id(a)] for a in tr_actions)
    vetoed = harness._guard_screen(root_ctx, cand_labels)
    legal_idx = [i for i in range(len(cand_actions)) if i not in vetoed]
    legal_pos = {full_i: pos for pos, full_i in enumerate(legal_idx)}
    admission = {legal_pos[i] for i in keep if i not in vetoed}
    return len(legal_idx), admission, keep


def process_root(ctx: Ctx, gctx, rec, entry):
    """处理一根，返回结果记录（valid 或 invalid＋原因码）。"""
    seat = entry["seat"]
    res = {
        "root_id": entry["root_id"], "game": entry["game"],
        "seed": entry["seed"], "step": entry["step"], "seat": seat,
        "key_uid": entry["key_uid"],
        "pi_r_given_g": entry["pi_r_given_g"],
        "pi_g": 1.0,
        "stratum": entry.get("stratum"),
        "settlement_decidable": bool(entry["settlement_decidable"]),
        "settlement_argmax": entry["settlement_argmax"],
        "cand_count_frozen": entry["cand_count"],
        "status": None, "invalid_reason": None,
    }
    t_root = time.monotonic()

    def invalid(reason, **extra):
        res["status"] = "invalid"
        res["invalid_reason"] = reason
        res.update(extra)
        res["wall_s_total"] = time.monotonic() - t_root
        return res

    obs_seat = rec["obs"][seat]
    a0 = rec["a0"]

    # 1. 触发复核：key_units 含 key_uid 且 stake 逐字合。
    keys = ctx.key_units(obs_seat, ctx.cfg)
    hit = [(float(s), u) for s, u in keys if u == entry["key_uid"]]
    if not hit or abs(hit[0][0] - float(entry["stake"])) > STAKE_TOL:
        return invalid("key_recheck_failed",
                       key_units_returned=[[float(s), u] for s, u in keys])

    harness = gctx["harnesses"][seat]
    cleanup = prepare_harness(harness, rec, ctx.feat_copy)
    try:
        root_ctx = harness._build_root(copy.deepcopy(obs_seat), a0)

        # 2. §6 准入核对（仅可辨根）。
        if entry["settlement_decidable"]:
            legal_n, admission, _keep = admission_legal_set(
                harness, root_ctx, a0)
            res["legal_cand_count_recheck"] = legal_n
            res["admission_legal_indices"] = sorted(admission)
            if entry["settlement_argmax"] not in admission:
                return invalid("admission_floor_violation")

        # 3. 叶差 margin：H=1 轮值复算（冻结 harness 机械）。
        t_m = time.monotonic()
        raw = harness._gen_candidates_raw(root_ctx, a0)
        if raw is None:
            res["leaf_margin"] = None
            res["margin_status"] = "no_candidates"
            res["leaf_values"] = {}
        else:
            cand_actions, cand_labels = raw
            vetoed = harness._guard_screen(root_ctx, cand_labels)
            rollout = harness._make_rollout(root_ctx)
            values = {}
            for i, act in enumerate(cand_actions):
                if i != 0 and i in vetoed:
                    continue
                ls = rollout(root_ctx.full, act, 1, entry["step"])
                values[i] = float(harness.leaf_adapter(ls))
            res["leaf_values"] = {str(i): v for i, v in values.items()}
            res["guard_vetoed"] = sorted(vetoed)
            non_a0 = [v for i, v in values.items() if i != 0]
            if non_a0:
                res["leaf_margin"] = max(non_a0) - values[0]
                res["margin_status"] = "ok"
            else:
                res["leaf_margin"] = None
                res["margin_status"] = "no_eligible_non_a0"
            # harness 侧 argmax（平局优先 A0）供与内核交叉。
            best_i = min(values, key=lambda i: (-values[i], i != 0, i)) \
                if values else 0
            res["harness_best_index"] = best_i
            res["harness_action"] = cand_actions[best_i]
        res["margin_wall_s"] = time.monotonic() - t_m
    finally:
        cleanup()

    # 4. A 臂决策：权威比较器（resync 包 agent 恢复 pre-call 后调用）。
    kagent = gctx["kernel_agent"]
    s1_verify3.restore_agent(kagent, rec["pre"][seat], ctx.feat_copy)
    before = kagent.cmp_stats.snapshot()
    t_d = time.monotonic()
    chosen = kagent(copy.deepcopy(obs_seat))
    decide_s = time.monotonic() - t_d
    after = kagent.cmp_stats.snapshot()
    aborts_delta = {}
    for k, v in after["abort_reasons"].items():
        d = v - before["abort_reasons"].get(k, 0)
        if d:
            aborts_delta[k] = d
    cmp_tel = {
        "calls_delta": after["calls"] - before["calls"],
        "completed_rounds_delta": (after["completed_rounds"]
                                   - before["completed_rounds"]),
        "changed_count_delta": (after["changed_count"]
                                - before["changed_count"]),
        "leaf_calls_delta": after["leaf_calls"] - before["leaf_calls"],
        "abort_reasons_delta": aborts_delta,
    }
    res["cmp_telemetry"] = cmp_tel
    res["decide_wall_s"] = decide_s
    res["a0_action"] = a0
    res["chosen_action"] = chosen
    res["changed"] = bool(chosen != a0)
    if res.get("harness_action") is not None:
        match = bool(res["harness_action"] == chosen)
        res["harness_kernel_action_match"] = match
        if not match:
            res["mismatch_explained_by_kernel_abort"] = bool(aborts_delta)

    # 5. 双臂续弈。
    arm_a0 = run_arm_s3(ctx, rec, a0)
    arm_chg = run_arm_s3(ctx, rec, chosen)
    res["arms"] = {"a0": arm_a0, "chosen": arm_chg}
    repro_ok = all(
        (a.get("root_call_reproduced") or {}).get("root_seat")
        and (a.get("root_call_reproduced") or {}).get("opp_seat")
        for a in (arm_a0, arm_chg))
    res["restore_reproduction_ok"] = bool(repro_ok)
    if not repro_ok:
        return invalid("restore_failed")
    if not (arm_a0.get("complete") and arm_chg.get("complete")):
        bad = [n for n, a in (("a0", arm_a0), ("chosen", arm_chg))
               if not a.get("complete")]
        return invalid("arm_incomplete:" + ",".join(
            f"{n}({(arm_a0 if n == 'a0' else arm_chg).get('incomplete_reason')})"
            for n in bad))
    res["status"] = "valid"
    res["wall_s_total"] = time.monotonic() - t_root
    return res


def load_done_ids(path, retry_invalid):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done[r["root_id"]] = r.get("status")
    if retry_invalid:
        done = {k: v for k, v in done.items() if v == "valid"}
    return done


def run_game(ctx: Ctx, game, entries, out_fh, done_ids, stats):
    """一局：基线重放采集＋在线逐根处理。entries 为该局入样根。"""
    seed = 986200 + game
    by_step = {}
    for e in entries:
        by_step.setdefault(e["step"], []).append(e)
    agents = [s1_verify3.make_fresh_agent(ctx.pack_path),
              s1_verify3.make_fresh_agent(ctx.pack_path)]
    gctx = {
        "harnesses": [real_env.RealEnvSearch(agents[0]),
                      real_env.RealEnvSearch(agents[1])],
        "kernel_agent": make_kernel_agent(ctx.resync_pack),
    }
    sim = ctx.FastCarbon()
    sim.reset(seed=seed)
    processed_steps = set()
    while sim.step_count < TERMINAL_STEP:
        step = sim.step_count
        todays_all = by_step.get(step)
        if not todays_all:
            obs = [sim.obs(0), sim.obs(1)]
            a = [agents[0](obs[0]), agents[1](obs[1])]
            sim.step(a[0], a[1])
            if ctx.terminal_fn(sim):
                break
            continue
        todays = [e for e in todays_all if e["root_id"] not in done_ids]
        obs = [sim.obs(0), sim.obs(1)]
        pre = post = sim_state = None
        if todays and not ctx.terminal_fn(sim):
            pre = [s1_verify3.snapshot_memory(agents[0], ctx.feat_copy),
                   s1_verify3.snapshot_memory(agents[1], ctx.feat_copy)]
            sim_state = sim.get_state()
        actions = [None, None]
        base_cost = [0.0, 0.0]
        for s in (0, 1):
            t0 = time.perf_counter()
            actions[s] = agents[s](obs[s])
            base_cost[s] = time.perf_counter() - t0
        if todays and pre is not None:
            post = [s1_verify3.snapshot_memory(agents[0], ctx.feat_copy),
                    s1_verify3.snapshot_memory(agents[1], ctx.feat_copy)]
            processed_steps.add(step)
            for e in todays:
                s = e["seat"]
                rec = {"game": game, "seed": seed, "seat": s,
                       "step": step, "obs": copy.deepcopy(obs),
                       "sim_state": sim_state, "pre": pre, "post": post,
                       "a0": copy.deepcopy(actions[s]),
                       "opp_a0": copy.deepcopy(actions[1 - s]),
                       "base_cost": base_cost[s]}
                try:
                    res = process_root(ctx, gctx, rec, e)
                except Exception as exc:
                    if isinstance(exc, RuntimeError) and "漂移" in str(exc):
                        raise  # 实现漂移＝管线完整性失败，整场停（§1.4 形态）
                    res = {"root_id": e["root_id"], "game": game,
                           "seed": seed, "step": step, "seat": s,
                           "pi_r_given_g": e["pi_r_given_g"], "pi_g": 1.0,
                           "stratum": e.get("stratum"),
                           "settlement_decidable": bool(
                               e["settlement_decidable"]),
                           "settlement_argmax": e["settlement_argmax"],
                           "status": "invalid",
                           "invalid_reason": f"exception:{exc!r}"}
                out_fh.write(json.dumps(res, ensure_ascii=False,
                                        default=str) + "\n")
                out_fh.flush()
                os.fsync(out_fh.fileno())
                done_ids[e["root_id"]] = res["status"]
                stats["roots_done"] += 1
                stats["walls"].append(res.get("wall_s_total"))
                print(f"    [{e['root_id']}] {res['status']}"
                      f"{'' if res['status'] == 'valid' else ' ' + str(res.get('invalid_reason'))}"
                      f" changed={res.get('changed')} "
                      f"margin={res.get('leaf_margin')} "
                      f"wall={res.get('wall_s_total', 0):.1f}s",
                      flush=True)
        sim.step(actions[0], actions[1])
        if ctx.terminal_fn(sim):
            break
    # 未到达的入样根（局提前灭绝等）→ invalid，不沉默。
    for step, es in by_step.items():
        for e in es:
            if e["root_id"] not in done_ids:
                res = {"root_id": e["root_id"], "game": game, "seed": seed,
                       "step": step, "seat": e["seat"],
                       "pi_r_given_g": e["pi_r_given_g"], "pi_g": 1.0,
                       "stratum": e.get("stratum"),
                       "settlement_decidable": bool(
                           e["settlement_decidable"]),
                       "settlement_argmax": e["settlement_argmax"],
                       "status": "invalid",
                       "invalid_reason": "root_not_reached_in_replay"}
                out_fh.write(json.dumps(res, ensure_ascii=False,
                                        default=str) + "\n")
                out_fh.flush()
                os.fsync(out_fh.fileno())
                done_ids[e["root_id"]] = "invalid"
                stats["roots_done"] += 1
                print(f"    [{e['root_id']}] invalid "
                      f"root_not_reached_in_replay", flush=True)


def main():
    ap = argparse.ArgumentParser(description="S3 双臂运行器")
    ap.add_argument("--sampled", required=True, help="入样根 jsonl")
    ap.add_argument("--games", required=True, help="局范围 A-B")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--shard", default=None, help="分片名（默认 games 串）")
    ap.add_argument("--pack", default=None)
    ap.add_argument("--resync-pack", default=RESYNC_PACK_DEFAULT)
    ap.add_argument("--arm-cap-s", type=float, default=1200.0)
    ap.add_argument("--root-ids", default=None,
                    help="仅跑指定 root_id（逗号分隔，冒烟/调试用）")
    ap.add_argument("--retry-invalid", action="store_true")
    ap.add_argument("--pilot", action="store_true",
                    help="先导模式：产出 pilot_cost.json 成本汇总")
    args = ap.parse_args()
    g0, g1 = (int(x) for x in args.games.split("-"))
    shard = args.shard or f"g{g0}_{g1}"
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"results_{shard}.jsonl")

    entries_by_game = {}
    with open(args.sampled, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if g0 <= e["game"] <= g1:
                entries_by_game.setdefault(e["game"], []).append(e)
    if args.root_ids:
        want = set(args.root_ids.split(","))
        entries_by_game = {
            g: [e for e in es if e["root_id"] in want]
            for g, es in entries_by_game.items()}
        entries_by_game = {g: es for g, es in entries_by_game.items() if es}
    total = sum(len(v) for v in entries_by_game.values())
    print(f"入样根（局 {g0}–{g1}）：{total} 根，"
          f"{len(entries_by_game)} 局", flush=True)

    pack_path = args.pack or fetch_pack()
    ctx = Ctx(pack_path, args.resync_pack, args.arm_cap_s)
    print(f"基线包 {pack_path}；resync 包验合毕（cba5923c／五源合）",
          flush=True)

    done_ids = load_done_ids(out_path, args.retry_invalid)
    stats = {"roots_done": 0, "walls": []}
    t_all = time.monotonic()
    with open(out_path, "a", encoding="utf-8") as out_fh:
        for game in sorted(entries_by_game):
            todo = [e for e in entries_by_game[game]
                    if e["root_id"] not in done_ids]
            print(f"== 局 {game}（seed {986200 + game}）：入样 "
                  f"{len(entries_by_game[game])} 根、待跑 {len(todo)} ==",
                  flush=True)
            if todo:
                run_game(ctx, game, entries_by_game[game], out_fh,
                         done_ids, stats)
    walls = [w for w in stats["walls"] if w is not None]
    summary = {"shard": shard, "games": [g0, g1],
               "roots_processed_this_run": stats["roots_done"],
               "wall_s_total_run": time.monotonic() - t_all}
    if walls:
        ws = sorted(walls)
        summary["per_root_wall_s"] = {
            "mean": sum(ws) / len(ws),
            "p95": ws[min(len(ws) - 1, int(0.95 * len(ws)))],
            "max": ws[-1], "n": len(ws)}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.pilot:
        cost_path = os.path.join(args.out, "pilot_cost.json")
        with open(cost_path, "w", encoding="utf-8") as fh:
            json.dump({**summary,
                       "caliber": "本机口径（成本读数以考场先导为准）",
                       "stop_line_s": 30.0,
                       "stop_line_exceeded_on_this_caliber": bool(
                           walls and summary["per_root_wall_s"]["mean"]
                           > 30.0)},
                      fh, ensure_ascii=False, indent=1)
        print(f"先导成本汇总：{cost_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
