#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§7.5 分析管线自查（冻结协议 §7.5；开测门 ①，出数前置）。

用 estimand.py 的同一套函数（与 analyze.py 同源）复算修订版两则
合成反例，逐字对目标值：
  1. R2-11（修订 §1.4）：A0 臂分差 [10, −1000]、改子臂分差
     [−10, −900] → 逐根 ΔU −0.50249998／+0.01005454、平均 ΔU
     **−0.24622272**（8 位小数逐字合）。
  2. R2-12（修订 §2.5）：局 A（100 根、ΔU 全 +0.01、全有效）＋
     局 B（1 根、ΔU −0.20、验证失败记无效）→ 点估计 **+0.01**、
     最坏界 **[−0.495, +0.505]**，并触发「缺失敏感、不可裁定」。
两则有一则不合 → 本脚本 FAIL（退出码 1），管线停用、先修代码。
证据落 selfcheck_75_evidence.json（输入、算得全精度值、目标、判定）。
"""

import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import estimand  # noqa: E402

OUT = os.path.join(_HERE, "selfcheck_75_evidence.json")


def _fmt(v, nd):
    return f"{v:.{nd}f}"


def case_r2_11():
    """分差 → 逐根 U → ΔU → 平均（走 estimand.u_of_margin 同一函数）。"""
    a0_margins = [10.0, -1000.0]
    chg_margins = [-10.0, -900.0]
    per_root = []
    for m_a0, m_chg in zip(a0_margins, chg_margins):
        du = estimand.u_of_margin(m_chg) - estimand.u_of_margin(m_a0)
        per_root.append(du)
    mean_du = sum(per_root) / len(per_root)
    targets = {"root1": "-0.50249998", "root2": "0.01005454",
               "mean": "-0.24622272"}
    got = {"root1": _fmt(per_root[0], 8), "root2": _fmt(per_root[1], 8),
           "mean": _fmt(mean_du, 8)}
    checks = {k: got[k] == targets[k] for k in targets}
    return {
        "case": "R2-11 （修订 §1.4）",
        "inputs": {"a0_margins": a0_margins, "chg_margins": chg_margins,
                   "note": "工作例输入为逐根逐臂终局分差 M；U 由 "
                           "estimand.u_of_margin 按 §1.1 公式算"},
        "computed_raw": {"per_root_delta_u": per_root, "mean_delta_u": mean_du,
                         "u_values": {
                             "U(10)": estimand.u_of_margin(10.0),
                             "U(-10)": estimand.u_of_margin(-10.0),
                             "U(-900)": estimand.u_of_margin(-900.0),
                             "U(-1000)": estimand.u_of_margin(-1000.0)}},
        "formatted": got, "targets": targets, "checks": checks,
        "pass": all(checks.values()),
    }


def case_r2_12():
    """局 A 全有效＋局 B 零有效根 → aggregate_games 同路复算。"""
    games = [
        {"game": 0, "pi_g": 1.0,
         "roots": [{"value": 0.01, "pi_r": 1.0, "valid": True}
                   for _ in range(100)]},
        {"game": 1, "pi_g": 1.0,
         "roots": [{"value": None, "pi_r": 1.0, "valid": False,
                    "reason": "validation_failed"}]},
    ]
    agg = estimand.aggregate_games(games)
    point = agg["point_estimate"]
    bounds = agg["worst_case_bounds"]
    got = {"point": _fmt(point, 2),
           "bound_lo": _fmt(bounds[0], 3), "bound_hi": _fmt(bounds[1], 3)}
    targets = {"point": "0.01", "bound_lo": "-0.495", "bound_hi": "0.505"}
    checks = {k: got[k] == targets[k] for k in targets}
    checks["downgrade_triggered"] = bool(
        agg["flags"]["missing_sensitive_not_adjudicable"])
    return {
        "case": "R2-12 （修订 §2.5）",
        "inputs": {"game_A": "100 根、ΔU 全 +0.01、全有效、π=1",
                   "game_B": "1 根、ΔU −0.20、验证失败记无效",
                   "note": "工作例输入为逐根 ΔU 与有效性；聚合走 "
                           "estimand.aggregate_games（与 analyze 同函数）"},
        "computed_raw": {"point_estimate": point,
                         "worst_case_bounds": bounds,
                         "flags": agg["flags"],
                         "zero_valid_games": agg["zero_valid_games"]},
        "formatted": got, "targets": targets, "checks": checks,
        "pass": all(checks.values()),
        "note_on_void_flag": "零有效根局占比 1/2 > 10%，§2.4 的作废标记"
                             "（void_for_adjudication）亦触发，随证据同报；"
                             "§7.5 第 2 则的判定要件为点估计＋最坏界＋降级"
                             "触发，本则以此三项为准。",
    }


def main():
    with open(os.path.join(_HERE, "estimand.py"), "rb") as fh:
        estimand_md5 = hashlib.md5(fh.read()).hexdigest()
    c1 = case_r2_11()
    c2 = case_r2_12()
    ok = c1["pass"] and c2["pass"]
    doc = {
        "protocol": "S3_ROOT_LIST_FREEZE_PROTOCOL.md §7.5",
        "estimand_py_md5": estimand_md5,
        "cases": [c1, c2],
        "verdict": "PASS" if ok else "FAIL",
        "rule": "两则有一则不合，分析管线停用、先修代码；真实数据的 "
                "ΔU 与界由同一管线（estimand.py）算出。",
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    print(f"R2-11: 算得平均 ΔU {c1['formatted']['mean']}（目标 "
          f"{c1['targets']['mean']}）→ {'合' if c1['pass'] else '不合'}")
    print(f"R2-12: 点估计 {c2['formatted']['point']}、最坏界 "
          f"[{c2['formatted']['bound_lo']}, {c2['formatted']['bound_hi']}]"
          f"（目标 +0.01／[−0.495, +0.505]）→ "
          f"{'合' if c2['pass'] else '不合'}")
    print(f"§7.5 自查：{doc['verdict']}（证据 {OUT}）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
