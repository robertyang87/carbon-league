#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 估计量核心（分析管线共享件；selfcheck_75 与 analyze 同源调用）。

口径（S3 附则修订 A2，commit 3dec813d，md5 dc95e2c2；冻结协议 §7）：
  - 逐根逐臂 U：U ＝ 0.5·WDL ＋ 0.5·sigmoid(M/1000)，M ＝ 本席终局分数
    − 对手终局分数（终局分数对已由运行器按 F01／F04 权威口径落值，
    本模块只消费分数对，不再做终局换算）。
  - ΔU_根 ＝ U_改子臂 − U_A0臂；H=0 根（记录带 h_zero 标）ΔU 定义 0。
  - 聚合（修订 §2.3）：局内入样有效根以 1/π_{r|g} Hájek 加权得局均值；
    跨局以 1/π_g Hájek 得局等权平均；CI 按局级聚类（局均值层
    influence-function 标准误＋t 分位）。
  - 缺失（修订 §2.4）：无效根记无效、绝不填 0；零有效根局不删除——
    点估计以有数据局计，最坏界对每个零有效根局分别赋局均值 −1／+1
    代入跨局式同报；界跨 0 → 降级「缺失敏感、不可裁定」；零有效根
    局占入样局 >10% → 对裁定目的作废（独立触发）。
  - 副尺：原始分差差（M_改子 − M_A0）同路聚合、单独标注，绝不混入主尺。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------- U 尺

def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def wdl_of_margin(m: float) -> float:
    if m > 0:
        return 1.0
    if m < 0:
        return 0.0
    return 0.5


def u_of_margin(m: float) -> float:
    """U ＝ 0.5·WDL ＋ 0.5·sigmoid(M/1000)（修订 §1.1 逐字）。"""
    return 0.5 * wdl_of_margin(m) + 0.5 * sigmoid(m / 1000.0)


def margin_of_scores(scores: Sequence[float], seat: int) -> float:
    """本席分差 M ＝ 本席终局分数 − 对手终局分数。"""
    return float(scores[seat]) - float(scores[1 - seat])


def delta_u_from_scores(scores_a0: Sequence[float],
                        scores_chg: Sequence[float], seat: int) -> float:
    """逐根 ΔU：双臂分数对 → 各臂 M → U → 差（分析层复算口径）。"""
    return (u_of_margin(margin_of_scores(scores_chg, seat))
            - u_of_margin(margin_of_scores(scores_a0, seat)))


def margin_diff_from_scores(scores_a0: Sequence[float],
                            scores_chg: Sequence[float], seat: int) -> float:
    """副尺：原始分差差 M_改子 − M_A0（与主尺分开报）。"""
    return (margin_of_scores(scores_chg, seat)
            - margin_of_scores(scores_a0, seat))


# ---------------------------------------------------------------- 聚合

def _t_crit(df: int) -> float:
    """95% t 分位；scipy 可用时用精确值，否则用正态 1.96 兜底。"""
    if df < 1:
        return float("nan")
    try:
        from scipy.stats import t as _t
        return float(_t.ppf(0.975, df))
    except Exception:
        return 1.96


def hajek_game_mean(items: List[Dict[str, Any]]) -> Optional[float]:
    """局内 Hájek 均值：items 为该局有效根 [{value, pi_r}]。
    无有效根返回 None（零有效根局由上层按最坏界处置，不在此填 0）。"""
    sw = 0.0
    sv = 0.0
    for it in items:
        w = 1.0 / float(it["pi_r"])
        sw += w
        sv += w * float(it["value"])
    if sw <= 0:
        return None
    return sv / sw


def aggregate_games(games: List[Dict[str, Any]]) -> Dict[str, Any]:
    """跨局聚合（主尺/副尺共用本函数，value 字段由调用方选定）。

    games: [{game, pi_g, roots: [{value, pi_r, valid, reason?}]}]
      - valid=False 的根不进点估计，只计数并进逐局缺失报告；
      - value=None 且 valid=True 视为内部错误，抛错停（不许静默）。
    返回点估计、局聚类 CI、最坏界、预注册失效标记与逐局缺失报告。
    """
    per_game = []
    zero_valid_games = []
    n_sampled_games = len(games)
    for g in games:
        roots = g["roots"]
        valid_items = []
        invalid_reasons: Dict[str, int] = {}
        n_invalid = 0
        for r in roots:
            if r.get("valid", True):
                if r.get("value") is None:
                    raise ValueError(
                        f"有效根缺 value（game={g['game']}）—— 管线错误，停")
                valid_items.append(r)
            else:
                n_invalid += 1
                reason = str(r.get("reason") or "unspecified")
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
        mean_g = hajek_game_mean(valid_items)
        entry = {
            "game": g["game"],
            "pi_g": float(g.get("pi_g", 1.0)),
            "n_sampled_roots": len(roots),
            "n_valid": len(valid_items),
            "n_invalid": n_invalid,
            "invalid_reasons": invalid_reasons,
            "game_mean": mean_g,
            "invalid_share_gt_50pct": (
                len(roots) > 0 and n_invalid > 0.5 * len(roots)),
            "zero_valid": mean_g is None,
        }
        per_game.append(entry)
        if mean_g is None:
            zero_valid_games.append(g["game"])

    data_games = [e for e in per_game if e["game_mean"] is not None]
    if not data_games:
        return {
            "point_estimate": None, "ci95": None, "se": None,
            "n_games_sampled": n_sampled_games, "n_games_with_data": 0,
            "per_game": per_game, "zero_valid_games": zero_valid_games,
            "worst_case_bounds": None,
            "flags": {"no_data_games": True,
                      "missing_sensitive_not_adjudicable": True,
                      "void_for_adjudication": True},
        }

    # 跨局 Hájek（局等权）：点估计＝Σ (1/π_g)·mean_g / Σ (1/π_g)。
    def _cross(entries, extra):
        """extra: [(pi_g, assigned_mean)] 追加局（最坏界用）。"""
        num = 0.0
        den = 0.0
        for e in entries:
            w = 1.0 / e["pi_g"]
            num += w * e["game_mean"]
            den += w
        for pi_g, val in extra:
            w = 1.0 / pi_g
            num += w * val
            den += w
        return num / den if den > 0 else None

    point = _cross(data_games, [])
    # 局级聚类标准误（influence function，Hájek 比率线性化）：
    # ψ_g ＝ (1/π_g)·(mean_g − point)；SE² ＝ Σψ² / (Σ1/π_g)² · G/(G−1)。
    G = len(data_games)
    den = sum(1.0 / e["pi_g"] for e in data_games)
    psi = [(1.0 / e["pi_g"]) * (e["game_mean"] - point) for e in data_games]
    if G > 1 and den > 0:
        se = math.sqrt(sum(p * p for p in psi)) / den * math.sqrt(G / (G - 1))
        crit = _t_crit(G - 1)
        ci = [point - crit * se, point + crit * se]
    else:
        se = None
        ci = None

    # 最坏界：每个零有效根局分别赋 −1／+1 代入跨局式（修订 §2.4）。
    if zero_valid_games:
        zg = [e for e in per_game if e["zero_valid"]]
        lower = _cross(data_games, [(e["pi_g"], -1.0) for e in zg])
        upper = _cross(data_games, [(e["pi_g"], 1.0) for e in zg])
        bounds = [lower, upper]
    else:
        bounds = [point, point]

    zero_frac = (len(zero_valid_games) / n_sampled_games
                 if n_sampled_games else 0.0)
    crosses = bounds is not None and bounds[0] < 0.0 < bounds[1]
    flags = {
        "worst_case_bounds_cross_zero": bool(crosses),
        "missing_sensitive_not_adjudicable": bool(crosses),
        "zero_valid_game_fraction": zero_frac,
        "void_for_adjudication": bool(zero_frac > 0.10),
        "no_data_games": False,
    }
    return {
        "point_estimate": point, "ci95": ci, "se": se,
        "n_games_sampled": n_sampled_games,
        "n_games_with_data": G,
        "per_game": per_game,
        "zero_valid_games": zero_valid_games,
        "worst_case_bounds": bounds,
        "flags": flags,
    }


# ---------------------------------------------------------------- 分带

MARGIN_BANDS = [
    ("le0", "≤0", None, 0.0),
    ("0_30", "0–30", 0.0, 30.0),
    ("30_60", "30–60", 30.0, 60.0),
    ("60_100", "60–100", 60.0, 100.0),
    ("100_150", "100–150", 100.0, 150.0),
    ("gt150", ">150", 150.0, None),
]
NO_MARGIN_KEY = "no_margin"
NO_MARGIN_LABEL = "无 margin（无非 A0 合格候选）"


def margin_band_of(margin: Optional[float]) -> str:
    """带界：≤0／(0,30]／(30,60]／(60,100]／(100,150]／>150。
    margin=None（无非 A0 合格候选）归 no_margin 单列。"""
    if margin is None:
        return NO_MARGIN_KEY
    if margin <= 0.0:
        return "le0"
    if margin <= 30.0:
        return "0_30"
    if margin <= 60.0:
        return "30_60"
    if margin <= 100.0:
        return "60_100"
    if margin <= 150.0:
        return "100_150"
    return "gt150"


def _weighted_mean_cluster_ci(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """根级加权均值（权 1/π_r）＋局聚类稳健 CI（归因分带表用）。

    items: [{value, pi_r, game}]。局聚类 influence：
    ψ_g ＝ Σ_{i∈g} w_i (v_i − mean)；SE ＝ sqrt(Σψ_g²)/Σw · sqrt(G/(G−1))。
    样本 <2 或单局时 CI 退化为 None 并注明。
    """
    if not items:
        return {"n": 0, "mean": None, "ci95": None, "se": None}
    sw = sum(1.0 / float(it["pi_r"]) for it in items)
    mean = sum((1.0 / float(it["pi_r"])) * float(it["value"])
               for it in items) / sw
    by_game: Dict[int, float] = {}
    for it in items:
        w = 1.0 / float(it["pi_r"])
        by_game[it["game"]] = by_game.get(it["game"], 0.0) \
            + w * (float(it["value"]) - mean)
    G = len(by_game)
    if G > 1:
        se = math.sqrt(sum(v * v for v in by_game.values())) / sw \
            * math.sqrt(G / (G - 1))
        crit = _t_crit(G - 1)
        ci = [mean - crit * se, mean + crit * se]
    else:
        se = None
        ci = None
    return {"n": len(items), "mean": mean, "ci95": ci, "se": se}


def band_table(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """叶差分带主表＋≤0 对照带校验（快路径裁定 9651f30b §二）。

    records: [{margin, changed, delta_u, pi_r, game, valid, root_id}]
      - 仅 valid 根进表；invalid 根在计数里单列（不进带内统计）；
      - 每带报：根数、changed 根数、全根 ΔU 均值/CI、changed 根
        ΔU 均值/CI（changed 根数为 0 时该格记「无样本」＝None，
        不补零）；
      - 对照校验：≤0 带（及 no_margin 行，两臂同为 A0）逐根 ΔU
        必须恒为 0；任一非零 → control_ok=False 并列出违例根，
        上层据此停报。
    """
    bands: Dict[str, List[Dict[str, Any]]] = {}
    n_invalid = 0
    for r in records:
        if not r.get("valid", True):
            n_invalid += 1
            continue
        bands.setdefault(margin_band_of(r.get("margin")), []).append(r)

    rows = []
    control_violations = []
    order = [k for k, _l, _a, _b in
             [(k, l, a, b) for k, l, a, b in MARGIN_BANDS]] + [NO_MARGIN_KEY]
    labels = {k: l for k, l, _a, _b in MARGIN_BANDS}
    labels[NO_MARGIN_KEY] = NO_MARGIN_LABEL
    for key in order:
        items = bands.get(key, [])
        changed_items = [r for r in items if r.get("changed")]
        all_stats = _weighted_mean_cluster_ci(
            [{"value": r["delta_u"], "pi_r": r["pi_r"], "game": r["game"]}
             for r in items])
        if changed_items:
            chg_stats = _weighted_mean_cluster_ci(
                [{"value": r["delta_u"], "pi_r": r["pi_r"],
                  "game": r["game"]} for r in changed_items])
            chg_cell = {"n": len(changed_items), "mean": chg_stats["mean"],
                        "ci95": chg_stats["ci95"], "se": chg_stats["se"],
                        "no_sample": False}
        else:
            chg_cell = {"n": 0, "mean": None, "ci95": None, "se": None,
                        "no_sample": True}
        is_control = key in ("le0", NO_MARGIN_KEY)
        if is_control:
            for r in items:
                if float(r["delta_u"]) != 0.0:
                    control_violations.append({
                        "root_id": r.get("root_id"), "band": key,
                        "delta_u": r["delta_u"], "changed": r.get("changed"),
                        "margin": r.get("margin")})
        rows.append({
            "band": key, "label": labels[key],
            "is_control_band": is_control,
            "n_roots": len(items),
            "n_changed": len(changed_items),
            "delta_u_all": {"n": all_stats["n"], "mean": all_stats["mean"],
                            "ci95": all_stats["ci95"], "se": all_stats["se"],
                            "no_sample": len(items) == 0},
            "delta_u_changed_only": chg_cell,
        })
    return {
        "bands": rows,
        "n_invalid_excluded": n_invalid,
        "control_check": {
            "bands_checked": ["le0", NO_MARGIN_KEY],
            "ok": len(control_violations) == 0,
            "violations": control_violations,
            "rule": "≤0 带与无 margin 行两臂同为 A0，逐根 ΔU 应恒为 0；"
                    "任一非零即管线对照失败、停报（裁定 9651f30b §二）",
        },
    }
