#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 分析层（冻结协议 §7.3/§7.4；修订 R2-11/R2-12；快路径裁定
9651f30b §二）。

消费双臂运行器 results_*.jsonl ＋抽样补充件入样根单：
  - 逐根 ΔU 由分数对经 estimand.delta_u_from_scores 复算（管线
    §7.5 自查同一 estimand.py）；H=0 根（记录带 h_zero 时）ΔU 定义 0；
    invalid 剔除并计数（逐局缺失报告、原因分布）。
  - 主估计量：Hájek 局等权（π_g=1、π_{r|g} 补充件）＋局聚类 CI；
    零有效根局按最坏界处理并标注（跨 0 → 「缺失敏感、不可裁定」；
    占比 >10% → 作废标记）。
  - 拆分：可辨根子总体、分层 low/mid/high、changed/未 changed 根
    ΔU 分布（毒性归因）、副尺（原始分差差，单独标注）。
  - **叶差分带主表＋对照带停报**：按 margin ≤0／0–30／30–60／
    60–100／100–150／>150（原始分差尺度）报根数、changed 根数、
    ΔU 均值与 CI（全根与仅 changed 两格；无 changed 根记「无样本」
    不补零）。≤0 带（与无 margin 行）两臂同为 A0、ΔU 应恒 0——
    任一非零 → verdict=CONTROL_FAIL_STOP，全部估计量降入
    diagnostics_not_reportable、停报（裁定 9651f30b §二硬要求）。
  - 先导报告项（--pilot）：单根实测成本（mean/p95/max）、无效率、
    逐局 ΔU。
用法：python3 analyze.py --results results_*.jsonl --sampled
sampled_roots.jsonl --out analysis.json [--pilot]
退出码：0＝OK；2＝对照带失败停报；1＝硬错误。
"""

import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import estimand  # noqa: E402


def load_results(paths):
    """results 记录按 root_id 去重（重跑 --retry-invalid 时后条为准）。"""
    recs = {}
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                recs[r["root_id"]] = r
    return recs


def load_sampled(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                e = json.loads(line)
                out[e["root_id"]] = e
    return out


def root_delta(rec):
    """有效根 → (ΔU, 分差差)；由分数对复算。"""
    seat = rec["seat"]
    a0 = rec["arms"]["a0"]["final_scores"]
    chg = rec["arms"]["chosen"]["final_scores"]
    if rec.get("h_zero"):
        return 0.0, estimand.margin_diff_from_scores(a0, chg, seat)
    return (estimand.delta_u_from_scores(a0, chg, seat),
            estimand.margin_diff_from_scores(a0, chg, seat))


def build_games(sampled, results, value_key="delta_u", subset=None):
    """按入样总体组局：[{game, pi_g, roots:[{value, pi_r, valid,
    reason}]}]。subset: 对入样根的过滤谓词（子总体/分层用）。"""
    by_game = {}
    for rid, e in sampled.items():
        if subset is not None and not subset(e):
            continue
        rec = results.get(rid)
        if rec is None:
            item = {"value": None, "pi_r": e["pi_r_given_g"],
                    "valid": False, "reason": "no_result_record"}
        elif rec.get("status") != "valid":
            item = {"value": None, "pi_r": e["pi_r_given_g"],
                    "valid": False,
                    "reason": rec.get("invalid_reason") or "invalid"}
        else:
            du, md = root_delta(rec)
            item = {"value": du if value_key == "delta_u" else md,
                    "pi_r": e["pi_r_given_g"], "valid": True}
        by_game.setdefault(e["game"], []).append(item)
    return [{"game": g, "pi_g": 1.0, "roots": rs}
            for g, rs in sorted(by_game.items())]


def dist_block(records, key):
    """changed/未 changed 分布块：加权均值＋局聚类 CI＋未加权分位。"""
    items = [{"value": r[key], "pi_r": r["pi_r"], "game": r["game"]}
             for r in records]
    stats = estimand._weighted_mean_cluster_ci(items)
    vals = sorted(r[key] for r in records)

    def q(p):
        if not vals:
            return None
        return vals[min(len(vals) - 1, int(p * len(vals)))]
    return {"n": len(records), "weighted_mean": stats["mean"],
            "ci95_cluster": stats["ci95"],
            "quantiles_unweighted": {"min": q(0.0), "p25": q(0.25),
                                     "median": q(0.5), "p75": q(0.75),
                                     "max": q(1.0)}}


def main():
    ap = argparse.ArgumentParser(description="S3 分析层")
    ap.add_argument("--results", nargs="+", required=True,
                    help="results jsonl（可带 glob）")
    ap.add_argument("--sampled", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    paths = []
    for pat in args.results:
        paths.extend(sorted(glob.glob(pat)) or [pat])
    results = load_results(paths)
    sampled = load_sampled(args.sampled)
    if not sampled:
        print("入样根单为空 —— 停", file=sys.stderr)
        return 1

    # ---- 逐根复算与记录组装
    band_records = []
    valid_records = []
    n_invalid_records = 0
    mismatch_unexplained = []
    cmp_calls_anomaly = 0
    for rid, e in sampled.items():
        rec = results.get(rid)
        if rec is None:
            continue
        if rec.get("status") != "valid":
            n_invalid_records += 1
            continue
        du, md = root_delta(rec)
        rr = {"root_id": rid, "game": e["game"], "pi_r": e["pi_r_given_g"],
              "margin": rec.get("leaf_margin"),
              "changed": bool(rec.get("changed")),
              "delta_u": du, "margin_diff": md, "valid": True}
        band_records.append(rr)
        valid_records.append(rr)
        if rec.get("harness_kernel_action_match") is False and not \
                rec.get("mismatch_explained_by_kernel_abort"):
            mismatch_unexplained.append(rid)
        tel = rec.get("cmp_telemetry") or {}
        if tel.get("calls_delta") not in (None, 1):
            cmp_calls_anomaly += 1

    bands = estimand.band_table(band_records)
    control_ok = bands["control_check"]["ok"]

    # ---- 聚合（主尺/副尺/子总体/分层）
    main_agg = estimand.aggregate_games(build_games(sampled, results))
    secondary_agg = estimand.aggregate_games(
        build_games(sampled, results, value_key="margin_diff"))
    decidable_agg = estimand.aggregate_games(build_games(
        sampled, results,
        subset=lambda e: e["settlement_decidable"]))
    strata_agg = {}
    game_stratum = {}
    for e in sampled.values():
        game_stratum[e["game"]] = e.get("stratum")
    for st in ("low", "mid", "high"):
        sub = build_games(sampled, results,
                          subset=lambda e, s=st: e.get("stratum") == s)
        if sub:
            strata_agg[st] = estimand.aggregate_games(sub)

    chg = [r for r in valid_records if r["changed"]]
    unchg = [r for r in valid_records if not r["changed"]]
    toxicity = {
        "changed_roots": dist_block(chg, "delta_u") if chg else
        {"n": 0, "note": "无样本"},
        "unchanged_roots": dist_block(unchg, "delta_u") if unchg else
        {"n": 0, "note": "无样本"},
    }

    estimates = {
        "main_delta_u": main_agg,
        "secondary_margin_diff_NOT_main_scale": {
            "point_estimate": secondary_agg["point_estimate"],
            "ci95": secondary_agg["ci95"],
            "note": "副尺（原始分差差），与主尺 U 分开读，不支撑 U 结论",
        },
        "decidable_subpopulation": decidable_agg,
        "strata": strata_agg,
        "toxicity_changed_vs_unchanged": toxicity,
    }

    pilot_block = None
    if args.pilot:
        walls = sorted(r.get("wall_s_total") for r in results.values()
                       if r.get("wall_s_total") is not None)
        n_sampled = len(sampled)
        n_invalid_total = sum(
            1 for rid in sampled
            if results.get(rid) is None
            or results[rid].get("status") != "valid")
        pilot_block = {
            "per_root_cost_s": (
                {"mean": sum(walls) / len(walls),
                 "p95": walls[min(len(walls) - 1, int(0.95 * len(walls)))],
                 "max": walls[-1], "n": len(walls)} if walls else None),
            "cost_caliber": "本机口径（考场先导为准）；停线 30 秒/根",
            "invalid_rate": (n_invalid_total / n_sampled
                             if n_sampled else None),
            "n_sampled": n_sampled, "n_invalid": n_invalid_total,
            "per_game_delta_u": [
                {"game": e["game"], "game_mean_delta_u": e["game_mean"],
                 "n_valid": e["n_valid"], "n_sampled_roots":
                 e["n_sampled_roots"]}
                for e in main_agg["per_game"]],
        }

    diagnostics = {
        "n_sampled_roots": len(sampled),
        "n_result_records": len(results),
        "n_valid": len(valid_records),
        "n_invalid_records": n_invalid_records,
        "harness_kernel_unexplained_mismatches": mismatch_unexplained,
        "cmp_calls_delta_not_1_count": cmp_calls_anomaly,
    }

    report = {
        "control_check": bands["control_check"],
        "margin_band_table": bands,
        "diagnostics": diagnostics,
    }
    if args.pilot:
        report["pilot"] = pilot_block
    if control_ok:
        report["verdict"] = "OK"
        report["estimates"] = estimates
    else:
        # 对照带失败：停报——估计量降入不可报区，仅留诊断。
        report["verdict"] = "CONTROL_FAIL_STOP"
        report["stop_note"] = (
            "≤0 对照带（或无 margin 行）出现非零 ΔU：两臂同为 A0 时 "
            "ΔU 应恒 0，管线对照失败。按裁定 9651f30b §二停报：下列 "
            "estimates 仅作诊断、不得引用；先定位管线再重跑。")
        report["diagnostics_not_reportable"] = {"estimates": estimates}

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)

    # ---- 控制台摘要
    print(f"入样 {len(sampled)} 根｜结果记录 {len(results)}｜有效 "
          f"{len(valid_records)}｜无效 {n_invalid_records}")
    print("叶差分带主表（ΔU 均值 [CI]；仅 changed 格无根时记无样本）：")
    for row in bands["bands"]:
        a = row["delta_u_all"]
        c = row["delta_u_changed_only"]
        a_s = ("无样本" if a["no_sample"] else
               f"{a['mean']:+.4f} {a['ci95']}")
        c_s = ("无样本" if c["no_sample"] else
               f"{c['mean']:+.4f} {c['ci95']}")
        print(f"  {row['label']:>28}: 根 {row['n_roots']:>5}｜changed "
              f"{row['n_changed']:>5}｜全根 ΔU {a_s}｜仅 changed ΔU {c_s}")
    if control_ok:
        print("对照带校验：≤0 带 ΔU 全零 —— 合")
        m = main_agg
        print(f"主估计量 ΔU ＝ {m['point_estimate']:+.6f}，CI95 "
              f"{m['ci95']}，最坏界 {m['worst_case_bounds']}，标记 "
              f"{m['flags']}")
        print(f"判定：OK（报告 {args.out}）")
        return 0
    print("！！对照带校验失败：非零 ΔU 违例 "
          f"{len(bands['control_check']['violations'])} 根 —— 停报"
          f"（CONTROL_FAIL_STOP，报告 {args.out}）")
    for v in bands["control_check"]["violations"][:10]:
        print(f"    违例 {v}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
