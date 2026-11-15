#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S1 电池 v2（SI03 修复）：诊断/正式双模式 + 注册断言闸 + 自描述证据包。

旧 s1_battery.py 的唯一 PASS 条件是 len(completed)>0：B 正例
expect_met 全灭也照样输出 BATTERY: PASS、退出 0。本版把正式判读
改为逐项注册断言，任一未达即非 0 退出并逐项列 FAIL/未验：

  A1 完成比较 > 0（completed_comparisons 总和与完成根数均 > 0）
  A2 正例全过（expect_changed=True 的构造例 expect_met 必须全 True）
  A3 反例不取（expect_changed=False 的构造例 expect_met 必须全 True）
  A4 选中精确最优（正典比较例 chosen_is_settlement_best 全 True；
     R2-05：分母＝正典名单，缺例/缺字段不许消失在分母外）
  A5 叶↔结算误差阈内（leaf_vs_settlement_max_abs_err 须 finite 且
     0≤v≤1e-6，逐正典例；R2-05：NaN/负误差不得过）
  A6 权威终局一致性（结算按权威终局口径产出：SI01 修复已落地，
     fixture 按权威终局重做——298 根逐候选独立真跑一步到终态
     step299 对账、297 根 H=2 推进冻结于 299 并断言零越界策略
     调用、299/300/灭绝根入口拒绝记账；R2-05：逐例核实质字段，
     不只信 meta 口径字符串）
  A7 最低根数/局数（根 ≥ 40、局 ≥ 3；R2-05：从 roots 逐根重算
     唯一根/局数，不信 summary）
  A8 中止与异常为 0（中止根、B 腿 abort_reason、守卫异常、构造例
     error/exception、续弈 exception 均为 0；R2-05：B 腿中止必扫，
     超时回退不得冒充成功负例）
  A9 证据自描述（源码 md5 集/包 md5/CONF 快照/fixture 身份齐备
     且与在案用例名一致；R2-05：核实质字段，不只信元数据 boolean）

模式：
  formal（默认）：断言全 PASS 才退出 0；FAIL 退出 2；未验退出 3。
  diagnostic：同采集同断言，但只记账不裁决，跑完退出 0（承
    addendum 的诚实口径，两模式分开命名，不许混签）。
  --replay <旧证据 JSON>：不跑实局，对既有证据数据按正式闸回放
    判读（旧包缺自描述字段/权威结算口径者，相应断言记未验）。

采集规模常量与种子口径同 s1_battery（SEED0/ROOTS_TARGET 等）。
正式实局口径（本版）：A 腿走 real_env.decide_at_root 公开入口
（SI01/SI02 修复版：唯一 deadline 贯穿全链、终局根入口拒绝）；
s1_battery.decide_instrumented 为 SI02 修复前插桩（0.05 s 预算
下限补齐、候选生成在时钟外），与修复后公开入口不同口径，不用
于正式采集。B 腿 fixture 按权威终局重做（见 A6）。预算口径：
部署预算 cfg BUDGET=0.7 s 在本 VM 不可达（全链实测约 2–3 s/根），
正式跑以显式声明的 VM 对照预算 RUN_BUDGET_S 采集，部署口径
复验归 S2；预算口径随证据包 budget_caliber 字段同行。证据包单
JSON 原子落盘（临时文件 + os.replace），含 run_id、源码 md5 集、
包 md5、CONF 快照、fixture 身份（构造参数 hash）、逐根明细、
逐项断言结果。

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
# SI01 修复已落地（real_env 权威终局：最后有效动作观察 step298、
# 终态 step299、推进冻结、终局根入口拒绝），B 腿 fixture 已按权威
# 口径重做（297/298/299/300×双席＋灭绝∩到时，见 run_b_case_auth /
# run_rejection_case），叶值与结算对账均在权威轴上，故标记权威。
SETTLEMENT_RULE = "authoritative_step299"
AUTHORITATIVE_SETTLEMENT_RULE = "authoritative_step299"

EXIT_PASS, EXIT_FAIL, EXIT_UNVERIFIED = 0, 2, 3

# ---------------- 运行预算口径（显式声明，不许静默改数） ----------------
# 部署口径：Echo4 包 CONF BUDGET=0.7 s（decide_at_root 的总预算）。
# 本 VM 共享 CPU 上全链实测约 2–3 s/根（基线步耗 alone 0.42–0.69 s），
# 0.7 s 在本机必然 deadline——正式跑以 VM 对照预算采集，部署口径
# 考位复验归 S2。运行预算经环境变量可调，取值与声明随证据包同行。
DEPLOYMENT_BUDGET_S = 0.7
RUN_BUDGET_S = float(os.environ.get("S1B2_RUN_BUDGET_S", "6.0"))
# 采集墙钟同属 VM 对照口径：b1 常量 COLLECT_WALL_S=330/HARD_WALL_S=
# 470 在其源注释即标「参考值，非部署口径」；本 VM 负载下全链约
# 3.5 s/根时 330 s 采不满注册规模（首趟实测 22/48 根、根序列与
# 首轮逐根同序，纯墙钟截断）。正式跑可经环境变量显式放宽，实值
# 随证据包 fixture_identity.collection_params 如实记录，不许静默。
COLLECT_WALL_S = os.environ.get("S1B2_COLLECT_WALL_S")
HARD_WALL_S = os.environ.get("S1B2_HARD_WALL_S")


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


# fixture 身份覆盖的全部构造参数（R2-05 P3）：旧版只取 case/
# expect_changed/spec，改 step 298→100 或 seat 0→1 身份不变；且
# items 里 bool(expect_changed) 把 None（拒绝例）与 False 压成同
# 值。本版逐键取原值、缺失键以哨兵标记——None/False/缺失三者两两
# 可区分，任一构造参数变化都改身份。
_IDENTITY_KEYS = ("case", "expect_changed", "kind", "step", "root_step",
                  "me", "seat", "key_uid", "geometry", "cash", "workers",
                  "carbon_cells", "spec", "design", "design_note")
_IDENTITY_ABSENT = "__ABSENT__"


def fixture_identity(cases_meta, seed0, collection_params):
    """cases_meta: 构造例定义 dict（含上列任一构造参数）。"""
    items = []
    for c in cases_meta:
        ident = {k: (c[k] if k in c else _IDENTITY_ABSENT)
                 for k in _IDENTITY_KEYS}
        items.append({"case": c.get("case"),
                      "expect_changed": c.get("expect_changed"),
                      "identity_sha256": _sha256_json(ident)})
    return {"seed0": seed0, "collection_params": collection_params,
            "part_b_cases": items}


def atomic_write_json(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ---------------- 正式断言（R2-05 加固） ----------------
# 正典用例名单：与 run_live 的构造一一对应（8 权威终局比较例＋
# B9 灭绝∩到时＋5 入口拒绝例）。判读器按此名单校验：缺例、重名、
# 名单外用例、缺必需字段一律显形，不许缺字段用例消失在分母外。
CANONICAL_PART_B = {}
for _step in (298, 297):
    for _seat in (0, 1):
        CANONICAL_PART_B[f"B_capture_step{_step}_seat{_seat}"] = ("compare", True)
        CANONICAL_PART_B[f"B_deposit_step{_step}_seat{_seat}"] = ("compare", False)
CANONICAL_PART_B["B9_extinct_time_coincide"] = ("b9", False)
for _step in (299, 300):
    for _seat in (0, 1):
        CANONICAL_PART_B[f"REJ_step{_step}_seat{_seat}"] = ("rejection", None)
CANONICAL_PART_B["REJ_extinct_step100_seat0"] = ("rejection", None)

_REQUIRED_FIELDS = {
    "compare": ("expect_changed", "expect_met", "abort_reason",
                "chosen_is_settlement_best",
                "leaf_vs_settlement_max_abs_err", "settlement_rule",
                "settlement_source", "terminal_freeze", "root_step",
                "me", "key_uid"),
    "b9": ("expect_changed", "expect_met", "abort_reason",
           "leaf_vs_settlement_max_abs_err", "f01_expected_leaf",
           "suicidal_terminal", "settlement_rule", "root_step",
           "me", "key_uid"),
    "rejection": ("kind", "expect_changed", "rejected",
                  "decide_returned_none", "settlement_rule",
                  "root_step", "me"),
}


def _is_md5(v):
    return (isinstance(v, str) and len(v) == 32
            and all(ch in "0123456789abcdef" for ch in v.lower()))


def evaluate_formal(summary, roots, part_b, verification, meta):
    """返回逐项断言 [{id, name, status, detail}]，status ∈ PASS/FAIL/未验。

    R2-05 加固口径：先整档校验（正典名单/唯一身份/必需字段）再计
    算判据；A8 扫 B 腿 abort_reason（超时回退不得冒充成功负例）；
    A5 误差须 finite 且 0<=v<=tol；A7 从 roots 逐根重算唯一根/局
    数（不信 summary）；A6/A9 核实质字段（逐例 settlement_rule/
    结算来源/冻结探针、源码 md5 集/包 md5/CONF 快照/fixture 身份
    与在案用例名一致），不只信口径字符串与元数据 boolean。
    meta: {settlement_rule, source_md5s, pack_md5, conf_snapshot,
    fixture_identity}（run_live/run_replay 由证据本体填充）。
    """
    A = []

    def add(aid, name, status, detail):
        A.append({"id": aid, "name": name, "status": status,
                  "detail": detail})

    # ---- 整档校验：正典名单＋唯一身份＋必需字段 ---------------------
    by_name, dup_names, unknown_names = {}, [], []
    for c in part_b:
        if not isinstance(c, dict) or "case" not in c:
            unknown_names.append(repr(c)[:40])
            continue
        nm = c["case"]
        if nm in by_name:
            dup_names.append(nm)
        else:
            by_name[nm] = c
        if nm not in CANONICAL_PART_B:
            unknown_names.append(nm)
    missing_cases = [n for n in CANONICAL_PART_B if n not in by_name]
    missing_fields = {}
    for nm, (kind, _exp) in CANONICAL_PART_B.items():
        c = by_name.get(nm)
        if c is not None:
            miss = [f for f in _REQUIRED_FIELDS[kind] if f not in c]
            if miss:
                missing_fields[nm] = miss
    structural = []
    if missing_cases:
        structural.append(f"正典用例缺失 {missing_cases}")
    if dup_names:
        structural.append(f"用例重名 {dup_names}")
    if unknown_names:
        structural.append(f"名单外用例 {unknown_names}")
    if missing_fields:
        structural.append(f"必需字段缺失 {missing_fields}")

    def canon(kind_want, expect_want):
        return [by_name[n] for n, (k, e) in CANONICAL_PART_B.items()
                if k in kind_want and e is expect_want and n in by_name]

    def case_bad(c):
        """比较例未达：异常在案、带中止回退、expect 未如实达成、
        或记录的 expect_changed 与正典不符。"""
        kind, exp = CANONICAL_PART_B[c["case"]]
        return ("error" in c or "exception" in c
                or c.get("abort_reason") is not None
                or c.get("expect_met") is not True
                or c.get("expect_changed") is not exp
                or c["case"] in missing_fields)

    total_cc = sum(r.get("completed_comparisons", 0) for r in roots)
    completed_roots = sum(1 for r in roots
                          if r.get("completed_rounds", 0) > 0)
    add("A1", "完成比较 > 0",
        "PASS" if (total_cc > 0 and completed_roots > 0) else "FAIL",
        f"completed_comparisons 总和={total_cc}，完成根={completed_roots}")

    pos_names = [n for n, (k, e) in CANONICAL_PART_B.items()
                 if k in ("compare", "b9") and e is True]
    pos = canon(("compare", "b9"), True)
    bad_pos = [c["case"] for c in pos if case_bad(c)]
    miss_pos = [n for n in pos_names if n not in by_name]
    add("A2", "正例 expect_met 全过",
        "FAIL" if (miss_pos or bad_pos) else "PASS",
        f"正典正例 {len(pos_names)} 例，未达：{bad_pos if bad_pos else '无'}"
        + (f"；缺失：{miss_pos}" if miss_pos else ""))

    neg_names = [n for n, (k, e) in CANONICAL_PART_B.items()
                 if k in ("compare", "b9") and e is False]
    neg = canon(("compare", "b9"), False)
    bad_neg = [c["case"] for c in neg if case_bad(c)]
    miss_neg = [n for n in neg_names if n not in by_name]
    add("A3", "反例不取（expect_met 全过）",
        "FAIL" if (miss_neg or bad_neg) else "PASS",
        f"正典反例 {len(neg_names)} 例，未达：{bad_neg if bad_neg else '无'}"
        + (f"；缺失：{miss_neg}" if miss_neg else ""))

    cmp_names = [n for n, (k, e) in CANONICAL_PART_B.items()
                 if k == "compare"]
    bad_best = []
    for nm in cmp_names:
        c = by_name.get(nm)
        if c is None or nm in missing_fields \
                or c.get("chosen_is_settlement_best") is not True:
            bad_best.append(nm)
    add("A4", "选中精确最优",
        "FAIL" if bad_best else "PASS",
        f"正典比较例 {len(cmp_names)} 例（分母固定，不随字段在场与否"
        f"增减），未达：{bad_best if bad_best else '无'}")

    err_names = [n for n, (k, e) in CANONICAL_PART_B.items()
                 if k in ("compare", "b9")]
    bad_err = []
    for nm in err_names:
        c = by_name.get(nm)
        if c is None or nm in missing_fields:
            bad_err.append((nm, "缺例/缺字段"))
            continue
        v = c.get("leaf_vs_settlement_max_abs_err")
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not np.isfinite(v) or v < 0 or v > LEAF_SETTLE_TOL:
            bad_err.append((nm, v))
    add("A5", "叶↔结算误差阈内",
        "FAIL" if bad_err else "PASS",
        f"正典 {len(err_names)} 例，tol={LEAF_SETTLE_TOL}，须 finite 且 "
        f"0<=v<=tol，超阈/无效：{bad_err if bad_err else '无'}")

    # A6：核实质字段，不只信 meta 口径字符串。
    a6_problems = []
    any_rule_field = any("settlement_rule" in c for c in by_name.values())
    for nm, (kind, _exp) in CANONICAL_PART_B.items():
        c = by_name.get(nm)
        if c is None:
            continue
        if c.get("settlement_rule") != AUTHORITATIVE_SETTLEMENT_RULE:
            a6_problems.append(
                f"{nm} settlement_rule={c.get('settlement_rule')!r}")
        if kind == "compare":
            want_src = ("independent_step_to_299"
                        if int(c.get("root_step", 0)) >= 298
                        else "rollout_terminal_frozen_299")
            if c.get("settlement_source") != want_src:
                a6_problems.append(
                    f"{nm} settlement_source={c.get('settlement_source')!r}"
                    f"（须 {want_src}）")
            tf = c.get("terminal_freeze")
            if not isinstance(tf, dict) or tf.get("ok") is not True \
                    or tf.get("h3_final_step") != real_env.TERMINAL_STEP:
                a6_problems.append(f"{nm} 终局冻结探针未合 {tf!r}")
        elif kind == "b9":
            st = c.get("suicidal_terminal")
            if not isinstance(st, dict) \
                    or st.get("step_count") != real_env.TERMINAL_STEP \
                    or st.get("terminal_kind") != "extinct":
                a6_problems.append(f"{nm} 灭绝终局字段未合 {st!r}")
            if c.get("f01_expected_leaf") != -102.0:
                a6_problems.append(
                    f"{nm} f01_expected_leaf={c.get('f01_expected_leaf')!r}")
        elif kind == "rejection":
            if c.get("rejected") is not True \
                    or c.get("decide_returned_none") is not True:
                a6_problems.append(f"{nm} 入口拒绝未如实记账")
    rule = meta.get("settlement_rule")
    if rule is not None and rule != AUTHORITATIVE_SETTLEMENT_RULE:
        a6_problems.append(f"meta 口径字符串 {rule!r} 与权威口径不符")
    if a6_problems:
        a6_status = "FAIL"
    elif missing_cases or not any_rule_field:
        a6_status = "未验"
    else:
        a6_status = "PASS"
    add("A6", "权威终局一致性", a6_status,
        ("；".join(a6_problems) if a6_problems else
         ("逐例实质字段核合（settlement_rule/结算来源/冻结探针/灭绝"
          "恒等/拒绝记账），权威口径 " + AUTHORITATIVE_SETTLEMENT_RULE)
         if a6_status == "PASS" else
         "正典用例缺失或无逐例结算字段在案，实质无从核（旧包回放）"))

    # A7：从 roots 逐根重算唯一根/局数，不信 summary。
    root_ids = [(r.get("game"), r.get("seat"), r.get("step"))
                for r in roots if isinstance(r, dict)]
    id_complete = len(root_ids) == len(roots) and all(
        v is not None for t in root_ids for v in t)
    uniq_roots = len(set(root_ids)) if id_complete else 0
    uniq_games = len({t[0] for t in root_ids}) if id_complete else 0
    add("A7", "最低根数/局数达标",
        "PASS" if (id_complete and uniq_roots >= MIN_ROOTS_FORMAL
                   and uniq_games >= MIN_GAMES_FORMAL) else "FAIL",
        f"逐根重算：唯一根 {uniq_roots}（须 ≥{MIN_ROOTS_FORMAL}），"
        f"局 {uniq_games}（须 ≥{MIN_GAMES_FORMAL}）；summary 自报根 "
        f"{summary.get('total_roots')}、局 {summary.get('games_played')}"
        + ("；根记录缺 game/seat/step 身份字段" if not id_complete else ""))

    abort_roots = [r for r in roots if r.get("abort_reason")]
    b_aborts = [(c["case"], c.get("abort_reason")) for c in part_b
                if isinstance(c, dict) and c.get("abort_reason") is not None]
    b_exc = [c["case"] for c in part_b
             if isinstance(c, dict) and ("error" in c or "exception" in c)]
    v_exc = [i for i, v in enumerate(verification)
             if isinstance(v, dict) and "exception" in v]
    guard_exc = summary.get("guard_exceptions_total")
    problems = []
    if summary.get("aborted_roots"):
        problems.append(f"中止根 {summary['aborted_roots']}")
    if abort_roots:
        problems.append(f"根级 abort_reason 非空 {len(abort_roots)}")
    if b_aborts:
        problems.append(f"B 腿中止 {b_aborts}")
    if guard_exc:
        problems.append(f"守卫异常 {guard_exc}")
    if b_exc:
        problems.append(f"构造例异常 {b_exc}")
    if v_exc:
        problems.append(f"续弈异常下标 {v_exc}")
    add("A8", "中止与异常计数为 0",
        "FAIL" if problems else "PASS",
        "；".join(problems) if problems else "中止/守卫异常/构造异常/续弈异常均 0")

    # A9：核实质字段，不只信元数据 boolean。
    src_md5s = meta.get("source_md5s")
    pack_md5 = meta.get("pack_md5")
    conf_snap = meta.get("conf_snapshot")
    fix_id = meta.get("fixture_identity")
    a9_detail_bits = []
    a9_ok = True
    a9_absent = False
    if not isinstance(src_md5s, dict) or not src_md5s \
            or not all(_is_md5(v) for v in src_md5s.values()):
        a9_ok, a9_absent = False, True
        a9_detail_bits.append("源码 md5 集缺失/格式不合")
    if not _is_md5(pack_md5):
        a9_ok, a9_absent = False, True
        a9_detail_bits.append("包 md5 缺失/格式不合")
    if not isinstance(conf_snap, dict) or not conf_snap:
        a9_ok, a9_absent = False, True
        a9_detail_bits.append("CONF 快照缺失")
    fix_names = None
    if isinstance(fix_id, dict) and isinstance(fix_id.get("part_b_cases"), list):
        fix_names = [it.get("case") for it in fix_id["part_b_cases"]
                     if isinstance(it, dict)]
    a9_mismatch = False
    if fix_names is None:
        a9_ok, a9_absent = False, True
        a9_detail_bits.append("fixture 身份缺失")
    elif sorted(fix_names) != sorted(by_name) \
            or len(fix_names) != len(set(fix_names)):
        # 字段在场但与在案用例对不上：实质不一致，FAIL 不作未验。
        a9_ok, a9_mismatch = False, True
        a9_detail_bits.append(
            f"fixture 身份用例名与在案用例不一致（fixture "
            f"{len(fix_names)} 名 vs 在案 {len(by_name)} 名）")
    if a9_ok:
        a9_status = "PASS"
    elif a9_mismatch:
        a9_status = "FAIL"
    else:
        a9_status = "未验"
    add("A9", "证据自描述齐备", a9_status,
        "源码 md5 集/包 md5/CONF 快照/fixture 身份（与在案用例名一"
        "致）须随证据包同行；" + ("本包实质字段齐备且一致" if a9_ok
                                else "；".join(a9_detail_bits)))

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


# ---------------- B 腿权威终局 fixture（SI01 重做） ----------------
def _craft_auth(FastCarbon, step, spec, carbon_cells=None):
    """按 spec 造任意 step 的状态（tests_s1_fix.craft 同法）：无树、
    碳图按 carbon_cells（默认全零）。返回 (sim, snap)。"""
    sim = FastCarbon()
    sim.reset(seed=b1.SEED0)
    snap = sim.get_state()
    snap["step"] = int(step)
    snap["cash"] = [float(spec["cash"][0]), float(spec["cash"][1])]
    carbon = [0.0] * 225
    for pos, val in (carbon_cells or {}).items():
        carbon[int(pos)] = float(val)
    snap["carbon"] = carbon
    snap["trees"] = {}
    snap["trees_dict"] = {}
    snap["tid_counter"] = [0, 0]
    w = {}
    wc = [0, 0]
    for wid, (p, pos, cargo, kind) in spec["workers"].items():
        w[wid] = {"player": p, "pos": int(pos), "carbon": float(cargo),
                  "type": kind}
        suffix = int(str(wid).rsplit("-", 1)[1])
        wc[p] = max(wc[p], suffix + 1)
    snap["workers"] = w
    snap["wid_counter"] = wc
    sim2 = FastCarbon()
    sim2.set_state(snap)
    assert sim2.step_count == step
    return sim2, snap


def _geom_spec(seat, kind, bases):
    """正/反例几何（口径同首轮 P2/N1，按席位镜像）。

    bases = (my_base, my_base_b, op_base, op_base_b, k_pos, far_pos)。
    capture（正例期望）：敌 collector(8) 站我方基地格、我方
    collector(44) 邻格，踏入捕获＋交付为期望动作。
    deposit（反例期望）：敌 collector(8) 在远格、我方 collector(44)
    邻基地格，A0 踏入交付 +44 唯一最优，期望 changed=False。
    """
    my_base, my_base_b, op_base, _op_base_b, k_pos, far_pos = bases
    me, op = seat, 1 - seat
    key = f"player-{me}-worker-0"
    workers = {
        f"player-{me}-worker-1": (me, my_base_b, 0, "COLLECTOR"),
        f"player-{me}-worker-2": (me, my_base_b, 0, "COLLECTOR"),
        f"player-{op}-worker-1": (op, op_base, 0, "COLLECTOR"),
    }
    if kind == "capture":
        workers[key] = (me, k_pos, 44, "COLLECTOR")
        workers[f"player-{op}-worker-0"] = (op, my_base, 8, "COLLECTOR")
    else:
        workers[key] = (me, k_pos, 44, "COLLECTOR")
        workers[f"player-{op}-worker-0"] = (op, far_pos, 8, "COLLECTOR")
    return {"cash": [100, 100], "workers": workers}, key


def run_b_case_auth(pack_path, FastCarbon, F, name, step, seat, spec,
                    key_uid, expect_changed, design_note):
    """权威终局构造例：fresh agent → begin_step(obs) → 基线 A0 →
    decide_at_root 正式判读 → 同根逐候选叶值与权威终态现金差对账。

    结算口径：step298 根（H=1）逐候选独立真跑一步到终态 step299
    的现金差（R2 双证口径）；step297 根（H=2）用 rollout 终态
    （推进冻结于 299、terminal_kind=time）的真实终态现金差，并以
    H=3 冻结探针断言零越界（act_b 调用数：298 根 0、297 根 1，
    终步恰 299）。冻结断言失败记 error（正式闸 A8）。
    """
    out = {"case": name, "design": design_note, "me": seat,
           "key_uid": key_uid, "expect_changed": expect_changed,
           "root_step": step, "settlement_rule": SETTLEMENT_RULE}
    _pack, agent = real_env.load_pack(pack_path)
    harness = real_env.RealEnvSearch(agent, collector=StatsCollector())
    harness.cfg["BUDGET"] = RUN_BUDGET_S
    horizon = 1 if step >= 298 else 2
    sim, snap = _craft_auth(FastCarbon, step, spec)
    obs = sim.obs(seat)
    harness.begin_step(obs)
    t0 = time.perf_counter()
    a0 = agent(obs)
    base_cost = time.perf_counter() - t0
    out["baseline_step_cost_s"] = base_cost
    out["a0_action"] = a0
    if not a0:
        out["error"] = "基线 agent 返回空动作（冷启动 _inner 异常兜底）"
        return out
    dec = harness.decide_at_root(obs, a0, base_cost, horizon=horizon,
                                 scenario_seed=b1.SEED0 + 900)
    if dec is None:
        out["error"] = "无候选根（key_units 未触发或去重后候选 <2）"
        return out
    out["changed"] = bool(dec.changed)
    out["chosen_index"] = dec.chosen_index
    out["abort_reason"] = dec.abort_reason
    out["expect_met"] = (bool(dec.changed) == expect_changed)
    # ---- 同根逐候选测量（确定性同候选集）：叶值＋权威结算 ----
    root = harness._build_root(obs, a0)
    gen = harness._generate_candidates(root, a0)
    if gen is None:
        out["error"] = "测量重建无候选（与 decide 不一致）"
        return out
    candidates, _labels, vetoed = gen
    out["guard_vetoed"] = sorted(vetoed)
    out["acts_o"] = root.acts_o
    rollout = harness._make_rollout(root)
    leaf_values, settlements = {}, {}
    freeze_bad = []
    for c in candidates:
        ls = rollout(root.full, c.action, horizon, b1.SEED0 + 900)
        leaf_values[c.index] = harness.leaf_adapter(ls)
        if ls.sim.step_count != real_env.TERMINAL_STEP:
            freeze_bad.append(f"候选{c.index} rollout 终步 "
                              f"{ls.sim.step_count}≠299")
        if step >= 298:
            s2 = FastCarbon()
            s2.set_state(snap)
            if seat == 0:
                s2.step(c.action, root.acts_o)
            else:
                s2.step(root.acts_o, c.action)
            if s2.step_count != real_env.TERMINAL_STEP:
                freeze_bad.append(f"候选{c.index} 独立结算终步 "
                                  f"{s2.step_count}≠299")
            settlements[c.index] = s2.cash[seat] - s2.cash[1 - seat]
        else:
            if ls.terminal_kind != "time":
                freeze_bad.append(f"候选{c.index} 终局类型 "
                                  f"{ls.terminal_kind}≠time")
            settlements[c.index] = (float(ls.sim.cash[seat])
                                    - float(ls.sim.cash[1 - seat]))
    out["leaf_values"] = {str(k): v for k, v in leaf_values.items()}
    out["settlements"] = {str(k): v for k, v in settlements.items()}
    out["settlement_source"] = ("independent_step_to_299"
                                if step >= 298
                                else "rollout_terminal_frozen_299")
    out["leaf_vs_settlement_max_abs_err"] = max(
        abs(leaf_values[i] - settlements[i]) for i in leaf_values)
    best_idx = max(settlements, key=lambda i: settlements[i])
    out["settlement_best_index"] = best_idx
    out["settlement_best_unique"] = (
        sum(1 for v in settlements.values()
            if v == settlements[best_idx]) == 1)
    out["chosen_is_settlement_best"] = (dec.chosen_index == best_idx)
    out["key_move_a0"] = a0.get(key_uid, "STAY")
    out["key_move_chosen"] = \
        candidates[dec.chosen_index].action.get(key_uid, "STAY")
    # ---- 终局冻结探针（R1 口径）：H=3、act_b 计数、终步恰 299 ----
    mod = harness.inx.mod
    orig_act_b = mod["act_b"]
    calls = {"n": 0}

    def counting_act_b(*a, **k):
        calls["n"] += 1
        return orig_act_b(*a, **k)

    mod["act_b"] = counting_act_b
    try:
        ls3 = rollout(root.full, candidates[0].action, 3, b1.SEED0 + 900)
    finally:
        mod["act_b"] = orig_act_b
    want_calls = 0 if step >= 298 else 1
    out["terminal_freeze"] = {
        "h3_final_step": ls3.sim.step_count,
        "act_b_calls": calls["n"], "expected_act_b_calls": want_calls,
        "ok": (ls3.sim.step_count == real_env.TERMINAL_STEP
               and calls["n"] == want_calls)}
    if not out["terminal_freeze"]["ok"]:
        freeze_bad.append(f"H=3 冻结探针 act_b={calls['n']}（期望 "
                          f"{want_calls}）、终步 {ls3.sim.step_count}")
    if freeze_bad:
        out["error"] = "终局冻结断言失败：" + "；".join(freeze_bad)
    return out


def run_b9_extinction(pack_path, FastCarbon, F):
    """灭绝∩到时重合例（R3 口径）：root298，我方唯一 collector
    （载货 60、cash 10 不可招募）踏入敌轻载（2）格——同一步灭绝
    且到时。自杀候选叶值须合 F01 恒等 −102（灭绝 pen −2 − 存活方
    带入步前现金 100）；比较器不得改子取自杀动作。"""
    name = "B9_extinct_time_coincide"
    me = 0
    key_uid = "player-0-worker-0"
    spec = {"cash": [10, 100], "workers": {
        key_uid: (0, 112, 60, "COLLECTOR"),
        "player-1-worker-0": (1, 113, 2, "COLLECTOR"),
        "player-1-worker-1": (1, 200, 0, "COLLECTOR")}}
    out = {"case": name, "me": me, "key_uid": key_uid,
           "expect_changed": False, "root_step": 298,
           "settlement_rule": SETTLEMENT_RULE,
           "design": "灭绝∩到时重合（R3 口径）：自杀候选叶值须合 F01 "
                     "恒等 −102，终局类型 extinct、终步 299"}
    _pack, agent = real_env.load_pack(pack_path)
    harness = real_env.RealEnvSearch(agent, collector=StatsCollector())
    harness.cfg["BUDGET"] = RUN_BUDGET_S
    sim, _snap = _craft_auth(FastCarbon, 298, spec,
                             carbon_cells={113: 100.0})
    obs = sim.obs(me)
    harness.begin_step(obs)
    t0 = time.perf_counter()
    a0 = agent(obs)
    base_cost = time.perf_counter() - t0
    out["baseline_step_cost_s"] = base_cost
    if not a0:
        out["error"] = "基线 agent 返回空动作"
        return out
    dec = harness.decide_at_root(obs, a0, base_cost, horizon=1,
                                 scenario_seed=b1.SEED0 + 900)
    out["decide_returned_none"] = dec is None
    if dec is not None:
        out["changed"] = bool(dec.changed)
        out["chosen_index"] = dec.chosen_index
        out["abort_reason"] = dec.abort_reason
    root = harness._build_root(obs, a0)
    slot = root.f.own_uid.index(key_uid)
    dests = F.DEST[112]
    move = None
    for m in range(5):
        if int(dests[m]) == 113:
            move = m
    if move is None:
        out["error"] = "DEST[112] 无到 113 的移动"
        return out
    y = root.y.copy()
    y[slot] = move
    suicidal = root.f.to_actions(y)
    out["suicidal_differs_from_a0"] = \
        suicidal.get(key_uid) != a0.get(key_uid)
    rollout = harness._make_rollout(root)
    ls = rollout(root.full, suicidal, 1, b1.SEED0 + 900)
    v = harness.leaf_adapter(ls)
    en = ls.sim.workers.get("player-1-worker-0")
    out["suicidal_leaf_value"] = v
    out["suicidal_terminal"] = {
        "step_count": ls.sim.step_count,
        "terminal_kind": ls.terminal_kind,
        "sim_terminal": bool(harness.inx.terminal_fn(ls.sim)),
        "key_worker_alive": key_uid in ls.sim.workers,
        "enemy_cargo": None if en is None else en.carbon}
    out["f01_expected_leaf"] = -102.0
    out["leaf_vs_settlement_max_abs_err"] = abs(v - (-102.0))
    identity_ok = (ls.sim.step_count == real_env.TERMINAL_STEP
                   and ls.terminal_kind == "extinct"
                   and out["leaf_vs_settlement_max_abs_err"]
                   <= LEAF_SETTLE_TOL
                   and out["suicidal_differs_from_a0"])
    out["expect_met"] = bool(identity_ok
                             and (dec is None or not dec.changed))
    if not identity_ok:
        out["error"] = "灭绝∩到时 F01 恒等/终局断言失败（见 suicidal_terminal）"
    return out


def run_rejection_case(pack_path, FastCarbon, name, step, seat, spec=None):
    """SI01 入口拒绝记账：step299/300 根与已灭绝根 → decide_at_root
    返回 None 且 terminal_roots_rejected +1，如实入证据
    （expect_changed=None，不入 A2/A3 正反例口径）。"""
    out = {"case": name, "me": seat, "expect_changed": None,
           "root_step": step, "kind": "terminal_rejection",
           "settlement_rule": SETTLEMENT_RULE}
    _pack, agent = real_env.load_pack(pack_path)
    harness = real_env.RealEnvSearch(agent, collector=StatsCollector())
    harness.cfg["BUDGET"] = RUN_BUDGET_S
    if spec is None:
        sim = FastCarbon()
        sim.reset(seed=b1.SEED0)
        snap = sim.get_state()
        snap["step"] = int(step)
        sim2 = FastCarbon()
        sim2.set_state(snap)
        assert sim2.step_count == step
        sim = sim2
    else:
        sim, _snap = _craft_auth(FastCarbon, step, spec)
    obs = sim.obs(seat)
    harness.begin_step(obs)
    a0 = agent(obs)
    before = harness.terminal_roots_rejected
    dec = harness.decide_at_root(obs, a0, baseline_step_cost_s=0.01)
    out["decide_returned_none"] = dec is None
    out["terminal_roots_rejected_delta"] = \
        harness.terminal_roots_rejected - before
    out["rejected"] = bool(dec is None
                           and out["terminal_roots_rejected_delta"] == 1)
    if not out["rejected"]:
        out["error"] = "已终局根未被入口拒绝（SI01 闸失效）"
    return out


# ---------------- 实局采集（正式口径：decide_at_root＋权威 fixture） ----------------
def run_live(mode):
    b1.T_START = time.time()
    if COLLECT_WALL_S is not None:
        b1.COLLECT_WALL_S = float(COLLECT_WALL_S)
    if HARD_WALL_S is not None:
        b1.HARD_WALL_S = float(HARD_WALL_S)
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
    pack_conf_budget = float(harnesses[0].cfg["BUDGET"])
    for h in harnesses:
        h.cfg["BUDGET"] = RUN_BUDGET_S
    print(f"== 预算口径声明：部署 BUDGET={pack_conf_budget}s（包 CONF）；"
          f"本 VM 全链约 2–3 s/根、部署预算在本机不可达，本趟以 VM "
          f"对照预算 {RUN_BUDGET_S}s 采集，部署口径考位复验归 S2 ==")

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
                harnesses[seat].begin_step(obs)
                t0 = time.perf_counter()
                act = agents[seat](obs)
                base_cost = time.perf_counter() - t0
                actions[seat] = act
                steps_scanned += 1
                if game_roots < b1.PER_GAME_ROOT_CAP \
                        and len(roots) < b1.ROOTS_TARGET:
                    leaf_before = harnesses[seat].leaf_adapter.calls
                    dec = harnesses[seat].decide_at_root(
                        obs, act, base_cost, horizon=1,
                        scenario_seed=b1.SEED0 + len(roots))
                    if dec is not None:
                        decs[seat] = dec
                        roots.append({
                            "game": g, "seat": seat, "step": obs["step"],
                            "chosen_index": dec.chosen_index,
                            "changed": bool(dec.changed),
                            "completed_rounds": dec.completed_rounds,
                            "completed_comparisons":
                                dec.completed_comparisons,
                            "deepest_completed_h": dec.deepest_completed_h,
                            "abort_reason": dec.abort_reason,
                            "elapsed_s": dec.elapsed_s,
                            "total_elapsed_s": dec.total_elapsed_s,
                            "phase_costs": dec.phase_costs,
                            "baseline_step_cost_s": base_cost,
                            "run_budget_total_s": RUN_BUDGET_S,
                            "leaf_calls":
                                harnesses[seat].leaf_adapter.calls
                                - leaf_before,
                            "n_guard_vetoed": None,
                            "leaf_values": {},
                            "detail_note": "正式入口 decide_at_root 口径："
                                "逐候选叶值/守卫否决明细不经公开入口暴露，"
                                "以 Decision 与 collector 遥测为准",
                        })
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

    print("== B. 权威终局构造例（SI01 重做：297/298×双席正反例"
          "＋灭绝∩到时＋299/300/灭绝根拒绝记账） ==")
    probe = FastCarbon()
    probe.reset(seed=b1.SEED0)
    base0 = probe.bases[0][0][1]
    base0b = probe.bases[0][1][1]
    base1 = probe.bases[1][0][1]
    base1b = probe.bases[1][1][1]

    def _adj_far(base, my_b, op_b, op_bb):
        adj = [c for c in range(225) if float(F.DIST[c, base]) == 1.0]
        k = adj[0]
        far = [c for c in range(225)
               if float(F.DIST[c, k]) == 2.0
               and float(F.DIST[c, base]) >= 2.0
               and c not in (base, my_b, op_b, op_bb)]
        return k, far[0]

    k0, far0 = _adj_far(base0, base0b, base1, base1b)
    k1, far1 = _adj_far(base1, base1b, base0, base0b)
    bases_by_seat = {0: (base0, base0b, base1, base1b, k0, far0),
                     1: (base1, base1b, base0, base0b, k1, far1)}
    case_defs = []
    for step in (298, 297):
        for seat in (0, 1):
            for kind, expect in (("capture", True), ("deposit", False)):
                spec, key = _geom_spec(seat, kind, bases_by_seat[seat])
                case_defs.append({
                    "case": f"B_{kind}_step{step}_seat{seat}",
                    "expect_changed": expect, "me": seat, "step": step,
                    "key_uid": key, "spec": spec,
                    "design_note":
                        "敌 collector(8) 站我方基地格、我方 collector(44) "
                        "邻格，踏入捕获＋交付为期望正例（首轮 P2 几何，"
                        "根步移至权威轴 297/298）"
                        if kind == "capture" else
                        "A0 踏入交付 +44 唯一最优，期望 changed=False"
                        "（首轮 N1 几何，根步移至权威轴 297/298）"})
    b9_spec = {"step": 298, "seat": 0, "geometry": "extinct_time",
               "cash": [10, 100],
               "workers": {"player-0-worker-0": [0, 112, 60, "COLLECTOR"],
                           "player-1-worker-0": [1, 113, 2, "COLLECTOR"],
                           "player-1-worker-1": [1, 200, 0, "COLLECTOR"]},
               "carbon_cells": {"113": 100.0}}
    rej_defs = [{"case": "B9_extinct_time_coincide",
                 "expect_changed": False, "spec": b9_spec}]
    for step in (299, 300):
        for seat in (0, 1):
            rej_defs.append({
                "case": f"REJ_step{step}_seat{seat}",
                "expect_changed": None,
                "spec": {"kind": "terminal_rejection", "step": step,
                         "seat": seat, "layout": "reset_standard"}})
    rej_defs.append({
        "case": "REJ_extinct_step100_seat0", "expect_changed": None,
        "spec": {"kind": "terminal_rejection", "step": 100, "seat": 0,
                 "cash": [100, 10],
                 "workers": {"player-0-worker-0": [0, 112, 40, "COLLECTOR"],
                             "player-0-worker-1": [0, 10, 0, "COLLECTOR"]}}})
    case_defs.extend(rej_defs)

    b_results = []
    for cd in case_defs[:8]:
        try:
            r = run_b_case_auth(pack_path, FastCarbon, F, cd["case"],
                                cd["step"], cd["me"], cd["spec"],
                                cd["key_uid"], cd["expect_changed"],
                                cd["design_note"])
            b_results.append(r)
            print(f"  {cd['case']}: changed={r.get('changed')} "
                  f"expect_met={r.get('expect_met')} err={r.get('error')}")
        except Exception as exc:
            b_results.append({"case": cd["case"],
                              "expect_changed": cd["expect_changed"],
                              "exception": repr(exc)})
            print(f"  {cd['case']} 异常：{exc!r}")
    try:
        r = run_b9_extinction(pack_path, FastCarbon, F)
        b_results.append(r)
        print(f"  B9_extinct_time_coincide: expect_met={r.get('expect_met')} "
              f"leaf={r.get('suicidal_leaf_value')} err={r.get('error')}")
    except Exception as exc:
        b_results.append({"case": "B9_extinct_time_coincide",
                          "expect_changed": False,
                          "exception": repr(exc)})
        print(f"  B9_extinct_time_coincide 异常：{exc!r}")
    extinct_spec = {"cash": [100, 10], "workers": {
        "player-0-worker-0": (0, 112, 40, "COLLECTOR"),
        "player-0-worker-1": (0, 10, 0, "COLLECTOR")}}
    for rd in rej_defs[1:]:
        step = rd["spec"]["step"]
        seat = rd["spec"]["seat"]
        spec = extinct_spec if rd["case"].startswith("REJ_extinct") else None
        try:
            r = run_rejection_case(pack_path, FastCarbon, rd["case"],
                                   step, seat, spec=spec)
            b_results.append(r)
            print(f"  {rd['case']}: rejected={r.get('rejected')} "
                  f"err={r.get('error')}")
        except Exception as exc:
            b_results.append({"case": rd["case"],
                              "expect_changed": None,
                              "exception": repr(exc)})
            print(f"  {rd['case']} 异常：{exc!r}")

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
        "total_guard_vetoed_candidates": None,
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
            "采集规模/种子口径同 s1_battery；判读为正式断言闸（SI03）；"
            "采集入口与 B 腿 fixture 已按 SI01/SI02 修复口径更新（见下）",
            f"结算口径 settlement_rule={SETTLEMENT_RULE}：298 根逐候选"
            "独立真跑一步到终态 299 对账、297 根 H=2 冻结于 299＋冻结"
            "探针、299/300/灭绝根入口拒绝记账（SI01 权威轴）",
            f"预算口径：部署 BUDGET={pack_conf_budget}s 在本 VM 不可达"
            f"（全链约 2–3 s/根），本趟以 VM 对照预算 RUN_BUDGET_S="
            f"{RUN_BUDGET_S}s 采集（decide_at_root 唯一 deadline 全链"
            "口径）；部署口径考位复验归 S2，本包计时不作部署证据",
            "A 腿走 decide_at_root 公开入口；s1_battery."
            "decide_instrumented 为 SI02 修复前插桩（0.05s 下限补齐、"
            "候选生成在时钟外），不同口径、未用于本趟",
            "逐根守卫否决数不经公开入口暴露，"
            "total_guard_vetoed_candidates 记 null（不冒充 0）；"
            "守卫异常计数 guard_exceptions_total 仍逐根入账",
            "scenario_seed 未消费：sweep 路径无种子轴，J=1 确定性",
            "叶①为原始分差尺度 affine 恒等直通",
            "采集墙钟为 VM 对照口径（S1B2_COLLECT_WALL_S/"
            "S1B2_HARD_WALL_S 声明，实值见 fixture_identity."
            "collection_params）；b1 原常量 330/470 为参考值",
            "墙钟为本 VM 共享 CPU 参考值，不作部署口径",
        ],
    }
    fix_id = fixture_identity(case_defs, b1.SEED0, collection_params)
    meta = {"settlement_rule": SETTLEMENT_RULE,
            "source_md5s": source_md5s(),
            "pack_md5": pack_md5,
            "conf_snapshot": conf_snapshot,
            "fixture_identity": fix_id}
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
        "budget_caliber": {
            "deployment_budget_s": pack_conf_budget,
            "run_budget_s": RUN_BUDGET_S,
            "caliber": "VM 对照口径（本机共享 CPU）",
            "note": "部署预算 0.7 s 在本机不可达：全链实测约 2–3 s/根、"
                    "基线步耗 alone 0.42–0.69 s；本趟以 run_budget_s 跑"
                    "采集，部署口径考位复验归 S2，本包计时数不作部署证据",
        },
        "fixture_identity": fix_id,
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
            "source_md5s": old.get("source_md5s"),
            "pack_md5": (old.get("pack") or {}).get("md5"),
            "conf_snapshot": old.get("conf_snapshot"),
            "fixture_identity": old.get("fixture_identity")}
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
