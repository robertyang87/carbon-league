#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S1 电池 v2（SI03 修复）：诊断/正式双模式 + 注册断言闸 + 自描述证据包。

旧 s1_battery.py 的唯一 PASS 条件是 len(completed)>0：B 正例
expect_met 全灭也照样输出 BATTERY: PASS、退出 0。本版把正式判读
改为逐项注册断言，任一未达即非 0 退出并逐项列 FAIL/未验：

  A1 完成比较 > 0（completed_comparisons 总和与完成根数均 > 0）
  A2 正例全过（expect_changed=True 的构造例 expect_met 必须全 True）
  A3 反例不取（expect_changed=False 的构造例 expect_met 必须全 True）
  A4 选中精确最优（每例 chosen_is_settlement_best 必须为 True）
  A5 叶↔结算误差阈内（leaf_vs_settlement_max_abs_err ≤ 1e-6，逐例）
  A6 权威终局一致性（结算须按权威终局口径产出；当前采集口径为
     SI01 修复前的 simulator step-to-300，恒记未验——终局修复落地
     并按权威口径重做 fixture 后才可能 PASS，不许自签）
  A7 最低根数/局数（根 ≥ 40、局 ≥ 3，注册电池规模口径）
  A8 中止与异常为 0（中止根、守卫异常、构造例 error/exception、
     续弈 exception 均为 0）
  A9 证据自描述（run_id/源码 md5 集/包 md5/CONF 快照/fixture 身份齐备）

模式：
  formal（默认）：断言全 PASS 才退出 0；FAIL 退出 2；未验退出 3。
  diagnostic：同采集同断言，但只记账不裁决，跑完退出 0（承
    addendum 的诚实口径，两模式分开命名，不许混签）。
  --replay <旧证据 JSON>：不跑实局，对既有证据数据按正式闸回放
    判读（旧包缺自描述字段/权威结算口径者，相应断言记未验）。

采集口径与 s1_battery 完全同源（import 其 decide_instrumented /
run_b_case / craft_state / verify_changed_root 与全部规模常量），
本版只换判读闸与证据包，不动采集语义。证据包单 JSON 原子落盘
（临时文件 + os.replace），含 run_id、源码 md5 集、包 md5、CONF
快照、fixture 身份（构造参数 hash）、逐根明细、逐项断言结果。

运行：
  python3 s1_battery2.py [--mode diagnostic] [--out PATH]
  python3 s1_battery2.py --replay s1_battery_roots.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.expanduser("~/workspace"))

# smoke_real_s1 在模块级解析 sys.argv[1] 为 int（其脚本用法），import
# 链（s1_battery -> smoke_real_s1）会与本脚本的 argparse 参数冲突；
# 导入期临时屏蔽 argv，导入后恢复（仅本文件内的导入技巧，不改他文件）。
_argv_saved = sys.argv
sys.argv = [_argv_saved[0]]
try:
    import s1_battery as b1                  # noqa: E402  采集口径同源
    from smoke_real_s1 import fetch_pack, pct  # noqa: E402
finally:
    sys.argv = _argv_saved
import real_env                              # noqa: E402
import pack_bridge as pb                     # noqa: E402
from telemetry import StatsCollector         # noqa: E402

# ---------------- 正式闸阈值（注册口径） ----------------
MIN_ROOTS_FORMAL = 40
MIN_GAMES_FORMAL = 3
LEAF_SETTLE_TOL = 1e-6
# 当前 B 结算由 FastCarbon 真跑一步到 step 300 产出，与 SI01 权威
# 终局口径（最后有效动作观察 step298、终态 step299）不一致；
# 核心组终局修复＋fixture 按权威口径重做前，此值不许改成权威。
SETTLEMENT_RULE = "simulator_step_to_300_pre_SI01_fix"
AUTHORITATIVE_SETTLEMENT_RULE = "authoritative_step299"

EXIT_PASS, EXIT_FAIL, EXIT_UNVERIFIED = 0, 2, 3


# ---------------- 证据包元数据 ----------------
def _file_md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def source_md5s():
    out = {}
    for fn in ("comparator.py", "real_env.py", "telemetry.py",
               "leaf_adapters.py", "pack_bridge.py", "s1_battery.py",
               "s1_battery2.py"):
        out[fn] = _file_md5(os.path.join(_HERE, fn))
    for name in pb.SOURCES:
        vpath = os.path.join(_HERE, "vendor", name + ".py")
        if os.path.exists(vpath):
            with open(vpath, "rb") as fh:
                out[f"vendor/{name}.py"] = pb.code_md5(name, fh.read())
    return out


def _sha256_json(obj) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def fixture_identity(cases_meta, seed0, collection_params):
    """cases_meta: [{case, expect_changed, spec|identity_fields}]。"""
    items = []
    for c in cases_meta:
        ident = {k: c[k] for k in ("case", "expect_changed")
                 if k in c}
        if "spec" in c:
            ident["spec"] = c["spec"]
        items.append({"case": c["case"],
                      "expect_changed": bool(c["expect_changed"]),
                      "identity_sha256": _sha256_json(ident)})
    return {"seed0": seed0, "collection_params": collection_params,
            "part_b_cases": items}


def atomic_write_json(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ---------------- 正式断言 ----------------
def evaluate_formal(summary, roots, part_b, verification, meta):
    """返回逐项断言 [{id, name, status, detail}]，status ∈ PASS/FAIL/未验。

    meta: {settlement_rule, evidence_meta_complete}
    """
    A = []

    def add(aid, name, status, detail):
        A.append({"id": aid, "name": name, "status": status,
                  "detail": detail})

    total_cc = sum(r.get("completed_comparisons", 0) for r in roots)
    completed_roots = sum(1 for r in roots
                          if r.get("completed_rounds", 0) > 0)
    add("A1", "完成比较 > 0",
        "PASS" if (total_cc > 0 and completed_roots > 0) else "FAIL",
        f"completed_comparisons 总和={total_cc}，完成根={completed_roots}")

    def cases_with(expect):
        return [c for c in part_b if c.get("expect_changed") is expect]

    def case_bad(c):
        return ("error" in c or "exception" in c
                or c.get("expect_met") is not True)

    pos = cases_with(True)
    bad_pos = [c["case"] for c in pos if case_bad(c)]
    add("A2", "正例 expect_met 全过",
        "FAIL" if (not pos or bad_pos) else "PASS",
        f"正例 {len(pos)} 例，未达：{bad_pos if bad_pos else '无'}"
        + ("（无正例在案）" if not pos else ""))

    neg = cases_with(False)
    bad_neg = [c["case"] for c in neg if case_bad(c)]
    add("A3", "反例不取（expect_met 全过）",
        "FAIL" if (not neg or bad_neg) else "PASS",
        f"反例 {len(neg)} 例，未达：{bad_neg if bad_neg else '无'}"
        + ("（无反例在案）" if not neg else ""))

    best_flags = [(c["case"], c.get("chosen_is_settlement_best"))
                  for c in part_b if "chosen_is_settlement_best" in c]
    bad_best = [n for n, v in best_flags if v is not True]
    if not best_flags:
        add("A4", "选中精确最优", "未验", "无任何构造例记录该字段")
    else:
        add("A4", "选中精确最优",
            "FAIL" if bad_best else "PASS",
            f"记录 {len(best_flags)} 例，未达：{bad_best if bad_best else '无'}")

    errs = [(c["case"], c["leaf_vs_settlement_max_abs_err"])
            for c in part_b if "leaf_vs_settlement_max_abs_err" in c]
    bad_err = [(n, v) for n, v in errs if v is None or v > LEAF_SETTLE_TOL]
    if not errs:
        add("A5", "叶↔结算误差阈内", "未验", "无任何构造例记录误差字段")
    else:
        add("A5", "叶↔结算误差阈内",
            "FAIL" if bad_err else "PASS",
            f"记录 {len(errs)} 例，tol={LEAF_SETTLE_TOL}，超阈：{bad_err}")

    rule = meta.get("settlement_rule")
    add("A6", "权威终局一致性",
        "PASS" if rule == AUTHORITATIVE_SETTLEMENT_RULE else "未验",
        f"在案结算口径={rule!r}；权威口径须为 "
        f"{AUTHORITATIVE_SETTLEMENT_RULE!r}（SI01：根/结算时间轴须按 "
        "step298 最后有效动作、终态 step299 重做后才可签）")

    total_roots = summary.get("total_roots", len(roots))
    games = summary.get("games_played")
    add("A7", "最低根数/局数达标",
        "PASS" if (total_roots >= MIN_ROOTS_FORMAL
                   and (games or 0) >= MIN_GAMES_FORMAL) else "FAIL",
        f"根 {total_roots}（须 ≥{MIN_ROOTS_FORMAL}），"
        f"局 {games}（须 ≥{MIN_GAMES_FORMAL}）")

    abort_roots = [r for r in roots if r.get("abort_reason")]
    b_exc = [c["case"] for c in part_b if "error" in c or "exception" in c]
    v_exc = [i for i, v in enumerate(verification)
             if isinstance(v, dict) and "exception" in v]
    guard_exc = summary.get("guard_exceptions_total")
    problems = []
    if summary.get("aborted_roots"):
        problems.append(f"中止根 {summary['aborted_roots']}")
    if abort_roots:
        problems.append(f"根级 abort_reason 非空 {len(abort_roots)}")
    if guard_exc:
        problems.append(f"守卫异常 {guard_exc}")
    if b_exc:
        problems.append(f"构造例异常 {b_exc}")
    if v_exc:
        problems.append(f"续弈异常下标 {v_exc}")
    add("A8", "中止与异常计数为 0",
        "FAIL" if problems else "PASS",
        "；".join(problems) if problems else "中止/守卫异常/构造异常/续弈异常均 0")

    add("A9", "证据自描述齐备",
        "PASS" if meta.get("evidence_meta_complete") else "未验",
        "run_id/源码 md5 集/包 md5/CONF 快照/fixture 身份须随证据包同行；"
        + ("本包齐备" if meta.get("evidence_meta_complete")
           else "来源证据缺上述字段（旧包回放）"))

    return A


def verdict_of(assertions):
    if any(a["status"] == "FAIL" for a in assertions):
        return "FAIL"
    if any(a["status"] == "未验" for a in assertions):
        return "未验"
    return "PASS"


def print_assertions(assertions, verdict):
    print("== 正式断言逐项 ==")
    for a in assertions:
        print(f"  [{a['status']}] {a['id']} {a['name']} —— {a['detail']}")
    print(f"正式判读：{verdict}")


def make_run_id(mode, payload) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"s1b2-{mode}-{ts}-{_sha256_json(payload)[:10]}"


# ---------------- 实局采集（口径同 s1_battery） ----------------
def run_live(mode):
    b1.T_START = time.time()
    print("== vendor md5 核验 ==")
    for k, v in real_env.verify_vendor().items():
        print(f"  {k}: {v}")
    pack_path = fetch_pack()
    pack_md5 = _file_md5(pack_path)
    pack_bytes = os.path.getsize(pack_path)
    print(f"== Echo4 包 md5 {pack_md5}（{pack_bytes} B）==")

    _p, agent0 = real_env.load_pack(pack_path)
    _p2, agent1 = real_env.load_pack(pack_path)
    agents = (agent0, agent1)
    collectors = (StatsCollector(), StatsCollector())
    harnesses = (real_env.RealEnvSearch(agent0, collector=collectors[0]),
                 real_env.RealEnvSearch(agent1, collector=collectors[1]))
    FastCarbon = harnesses[0].inx.FastCarbon
    F = harnesses[0].inx.F
    conf_snapshot = {str(k): (v if isinstance(v, (int, float, str, bool))
                              or v is None else str(v))
                     for k, v in harnesses[0].cfg.items()}

    roots, saved_verify = [], []
    games_played = 0
    steps_scanned = 0

    print("== A. 自然根电池（采集口径同 s1_battery） ==")
    for g in range(b1.MAX_GAMES):
        if len(roots) >= b1.ROOTS_TARGET and games_played >= b1.MIN_GAMES:
            break
        if b1.wall() > b1.COLLECT_WALL_S and games_played >= b1.MIN_GAMES:
            print(f"  采集墙钟上限 {b1.COLLECT_WALL_S}s 到，停开新局")
            break
        games_played += 1
        sim = FastCarbon()
        sim.reset(seed=b1.SEED0 + g)
        game_roots = 0
        step = 0
        while sim.step_count < 300 and step < b1.MAX_STEPS_PER_GAME:
            if len(roots) >= b1.ROOTS_TARGET and games_played >= b1.MIN_GAMES:
                break
            if game_roots >= b1.PER_GAME_ROOT_CAP:
                break
            if b1.wall() > b1.COLLECT_WALL_S:
                break
            step += 1
            actions, decs, details = [None, None], [None, None], [None, None]
            for seat in (0, 1):
                obs = sim.obs(seat)
                harnesses[seat].begin_step()
                t0 = time.perf_counter()
                act = agents[seat](obs)
                base_cost = time.perf_counter() - t0
                actions[seat] = act
                steps_scanned += 1
                if game_roots < b1.PER_GAME_ROOT_CAP \
                        and len(roots) < b1.ROOTS_TARGET:
                    dec, detail = b1.decide_instrumented(
                        harnesses[seat], obs, act, base_cost, horizon=1,
                        scenario_seed=b1.SEED0 + len(roots))
                    if dec is not None:
                        decs[seat] = dec
                        details[seat] = detail
                        roots.append(b1.root_record(
                            g, seat, obs["step"], dec, detail, base_cost))
                        game_roots += 1
            for seat in (0, 1):
                if decs[seat] is not None and decs[seat].changed \
                        and sim.step_count >= b1.VERIFY_MIN_STEP \
                        and len(saved_verify) < b1.VERIFY_MAX_ROOTS:
                    saved_verify.append({
                        "game": g, "seat": seat, "step": sim.step_count,
                        "snap": sim.get_state(),
                        "a0": actions[seat], "chosen": decs[seat].action,
                        "other": actions[1 - seat], "FastCarbon": FastCarbon})
            sim.step(actions[0], actions[1])
        print(f"  局 {g} 完：本局根 {game_roots}，累计 {len(roots)}，"
              f"墙钟 {b1.wall():.0f}s")

    print("== B. 精确终局正反例（fixture 同 s1_battery；结算口径见 A6） ==")
    probe = FastCarbon()
    probe.reset(seed=b1.SEED0)
    base0 = probe.bases[0][0][1]
    base0b = probe.bases[0][1][1]
    base1 = probe.bases[1][0][1]
    adj = [c for c in range(225) if float(F.DIST[c, base0]) == 1.0]
    k_pos = adj[0]
    far2 = [c for c in range(225)
            if float(F.DIST[c, k_pos]) == 2.0
            and float(F.DIST[c, base0]) >= 2.0
            and c not in (base0, base0b, base1, probe.bases[1][1][1])]
    e_pos_n1 = far2[0]
    extras_me = {"player-0-worker-1": (0, base0b, 0, "COLLECTOR"),
                 "player-0-worker-2": (0, base0b, 0, "COLLECTOR")}
    extra_en = {"player-1-worker-1": (1, base1, 0, "COLLECTOR")}
    case_defs = [
        {"case": "P2_capture_deposit", "expect_changed": True, "me": 0,
         "key_uid": "player-0-worker-0",
         "spec": {"cash": [100, 100], "workers": {
             "player-0-worker-0": (0, k_pos, 44, "COLLECTOR"),
             "player-1-worker-0": (1, base0, 8, "COLLECTOR"),
             **extras_me, **extra_en}},
         "design_note": "敌 collector(8) 站我方基地格，我方 collector(44) 邻格；"
                        "踏入捕获后交付为期望正例（同 s1_battery P2）。"},
        {"case": "P1_light_capture", "expect_changed": True, "me": 0,
         "key_uid": "player-0-worker-0",
         "spec": {"cash": [100, 100], "workers": {
             "player-0-worker-0": (0, k_pos, 12, "COLLECTOR"),
             "player-1-worker-0": (1, base0, 44, "COLLECTOR"),
             **extras_me, **extra_en}},
         "design_note": "敌 collector(44) 站我方基地格，我方 collector(12) 邻格；"
                        "踏入捕获交付为期望正例（同 s1_battery P1）。"},
        {"case": "N1_deposit_optimal", "expect_changed": False, "me": 0,
         "key_uid": "player-0-worker-0",
         "spec": {"cash": [100, 100], "workers": {
             "player-0-worker-0": (0, k_pos, 44, "COLLECTOR"),
             "player-1-worker-0": (1, e_pos_n1, 8, "COLLECTOR"),
             **extras_me, **extra_en}},
         "design_note": "A0 踏入交付 +44 唯一最优，期望 changed=False"
                        "（同 s1_battery N1）。"},
    ]
    b_results = []
    for cd in case_defs:
        try:
            r = b1.run_b_case(pack_path, FastCarbon, F, cd["case"],
                              cd["spec"], me=cd["me"],
                              key_uid=cd["key_uid"],
                              expect_changed=cd["expect_changed"],
                              design_note=cd["design_note"])
            b_results.append(r)
            print(f"  {cd['case']}: changed={r.get('changed')} "
                  f"expect_met={r.get('expect_met')} err={r.get('error')}")
        except Exception as exc:
            b_results.append({"case": cd["case"],
                              "expect_changed": cd["expect_changed"],
                              "exception": repr(exc)})
            print(f"  {cd['case']} 异常：{exc!r}")

    verifications = []
    for saved in saved_verify:
        if b1.wall() > b1.HARD_WALL_S - 60:
            print("  墙钟不足，事后验证跳过")
            break
        try:
            verifications.append(b1.verify_changed_root(pack_path, saved))
        except Exception as exc:
            verifications.append({"game": saved["game"],
                                  "seat": saved["seat"],
                                  "step": saved["step"],
                                  "exception": repr(exc)})

    elapsed = [r["elapsed_s"] for r in roots]
    completed = [r for r in roots if r["completed_rounds"] > 0]
    aborted = [r for r in roots if r["completed_rounds"] == 0]
    abort_dist = {}
    for r in aborted:
        abort_dist[r["abort_reason"]] = abort_dist.get(r["abort_reason"], 0) + 1
    depth_hist = {}
    for r in completed:
        d = str(r["deepest_completed_h"])
        depth_hist[d] = depth_hist.get(d, 0) + 1
    collection_params = {
        "target": b1.ROOTS_TARGET, "max_games": b1.MAX_GAMES,
        "per_game_cap": b1.PER_GAME_ROOT_CAP,
        "max_steps_per_game": b1.MAX_STEPS_PER_GAME,
        "collect_wall_s": b1.COLLECT_WALL_S, "hard_wall_s": b1.HARD_WALL_S,
    }
    summary = {
        "games_played": games_played,
        "steps_scanned": steps_scanned,
        "total_roots": len(roots),
        "completed_roots": len(completed),
        "aborted_roots": len(aborted),
        "abort_reasons": abort_dist,
        "changed_count": sum(1 for r in roots if r["changed"]),
        "total_guard_vetoed_candidates":
            sum(r["n_guard_vetoed"] for r in roots),
        "guard_exceptions_total":
            sum(h._guard_exceptions for h in harnesses),
        "total_leaf_calls": sum(r["leaf_calls"] for r in roots),
        "total_completed_comparisons":
            sum(r["completed_comparisons"] for r in roots),
        "elapsed_s_dist": {
            "mean": float(np.mean(elapsed)) if elapsed else None,
            "p50": pct(elapsed, 50) if elapsed else None,
            "p95": pct(elapsed, 95) if elapsed else None,
            "max": float(np.max(elapsed)) if elapsed else None,
        },
        "depth_hist": depth_hist,
        "collector_snapshots": [c.snapshot() for c in collectors],
        "wall_s_total": b1.wall(),
        "notes": [
            "采集与判读口径同 s1_battery；本版只换正式闸与证据包（SI03）",
            f"结算口径 settlement_rule={SETTLEMENT_RULE}：A6 恒未验，"
            "SI01 终局修复并按权威口径重做 fixture 后另跑正式模式",
            "scenario_seed 未消费：sweep 路径无种子轴，J=1 确定性",
            "叶①为原始分差尺度 affine 恒等直通",
            "墙钟为本 VM 共享 CPU 参考值，不作部署口径",
        ],
    }
    meta = {"settlement_rule": SETTLEMENT_RULE,
            "evidence_meta_complete": True}
    assertions = evaluate_formal(summary, roots, b_results,
                                 verifications, meta)
    verdict = verdict_of(assertions)
    doc = {
        "evidence_format": "s1_battery2/1",
        "run_id": make_run_id(mode, {"summary": summary,
                                     "assertions": assertions}),
        "mode": mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_md5s": source_md5s(),
        "pack": {"md5": pack_md5, "bytes": pack_bytes,
                 "path_in_repo": "code/packs/echo4_int8.py"},
        "conf_snapshot": conf_snapshot,
        "fixture_identity": fixture_identity(case_defs, b1.SEED0,
                                             collection_params),
        "settlement_rule": SETTLEMENT_RULE,
        "summary": summary, "roots": roots, "part_b": b_results,
        "verification": verifications,
        "assertions": assertions, "verdict": verdict,
    }
    return doc, verdict


# ---------------- 回放判读（不跑实局） ----------------
def run_replay(old_path):
    with open(old_path, encoding="utf-8") as fh:
        old = json.load(fh)
    summary = old.get("summary", {})
    roots = old.get("roots", [])
    part_b = old.get("part_b", [])
    verification = old.get("verification", [])
    meta = {"settlement_rule": old.get("settlement_rule"),
            "evidence_meta_complete": False}
    assertions = evaluate_formal(summary, roots, part_b,
                                 verification, meta)
    verdict = verdict_of(assertions)
    doc = {
        "evidence_format": "s1_battery2/1",
        "run_id": make_run_id("replay", {"assertions": assertions}),
        "mode": "replay",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "replay_of": {"file": os.path.basename(old_path),
                      "md5": _file_md5(old_path)},
        "source_md5s": source_md5s(),
        "pack": {"md5": real_env.EXPECTED_PACK_MD5,
                 "note": "旧证据自报包身份（本趟未重跑实局、未复验包文件）"},
        "conf_snapshot": None,
        "fixture_identity": fixture_identity(
            [{"case": c.get("case"), "expect_changed": c.get("expect_changed"),
              "spec": {"design": c.get("design"), "me": c.get("me"),
                       "key_uid": c.get("key_uid")}}
             for c in part_b],
            seed0=None,
            collection_params={"note": "旧证据未自带采集参数结构，"
                                       "见其 summary.notes"}),
        "settlement_rule": old.get("settlement_rule"),
        "summary": summary, "roots": roots, "part_b": part_b,
        "verification": verification,
        "assertions": assertions, "verdict": verdict,
        "notes": ["回放判读：对旧证据数据按 v2 正式闸重判，不跑实局；"
                  "旧包缺 CONF 快照/源码 md5/权威结算口径，A6/A9 记未验"],
    }
    return doc, verdict


def main():
    ap = argparse.ArgumentParser(description="S1 电池 v2（正式闸）")
    ap.add_argument("--mode", choices=("formal", "diagnostic"),
                    default="formal")
    ap.add_argument("--replay", metavar="JSON",
                    help="对既有证据 JSON 回放正式判读，不跑实局")
    ap.add_argument("--out", metavar="PATH", help="证据包落盘路径")
    args = ap.parse_args()

    if args.replay:
        doc, verdict = run_replay(args.replay)
        out = args.out or os.path.join(_HERE,
                                       "s1_battery2_replay_evidence.json")
        atomic_write_json(out, doc)
        print(f"回放判读证据包：{out}（run_id {doc['run_id']}）")
        print_assertions(doc["assertions"], verdict)
        print("REPLAY VERDICT:", verdict)
        return {"PASS": EXIT_PASS, "FAIL": EXIT_FAIL,
                "未验": EXIT_UNVERIFIED}[verdict]

    doc, verdict = run_live(args.mode)
    out = args.out or os.path.join(_HERE, "s1_battery2_evidence.json")
    atomic_write_json(out, doc)
    print(f"证据包：{out}（run_id {doc['run_id']}）")
    print_assertions(doc["assertions"], verdict)
    if args.mode == "diagnostic":
        print("DIAGNOSTIC: 跑完记账，正式判读（见上）不作裁决，退出 0")
        return 0
    print("BATTERY2 FORMAL VERDICT:", verdict)
    return {"PASS": EXIT_PASS, "FAIL": EXIT_FAIL,
            "未验": EXIT_UNVERIFIED}[verdict]


if __name__ == "__main__":
    sys.exit(main())
