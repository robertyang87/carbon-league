#!/usr/bin/env python3
"""A/B tables + mechanical verdict stamping (PREREG_VALUE signed text).

Test A (calibration): per checkpoint bias = mean(vhat - U),
MAE = mean|vhat - U|, Pearson, Spearman; overall + seat{0,1} x
step-band{200-249, 250-299} splits. Verdict lines (Delta = 95 - 74,
bias tracked as |bias|):
  - DAE = MAE_95 - MAE_74, DB = |bias_95| - |bias_74|.
  - A metric 'fires' if its Delta >= 0.020. Step attribution per
    metric m over the checkpoint sequence: head-swap group =
    (m78 - m74) + (m79 - m78); r19 group = (m82 - m79) + (m85 - m82)
    [telescopes to m85 - m79; with ckpt_82's vhat artifact missing the
    two single steps are unobservable but their signed sum is exact].
    Group share = group / Delta_m; share >= 40% stamps the group
    label for that metric.
  - A verdict: any firing metric with a group label ->
    '指认（换头步）' and/or '指认（r19 步）'; firing but no group >= 40%
    -> '中间（退化成立、步归因不成立）'; both Deltas <= 0.005 ->
    '证伪'; otherwise '无判定（量级未达指认门且未达证伪门）' with the
    deltas recorded (unsigned gap in the preregistration, reported
    as-is, no line invented).

Test B (sign localization): A_act(root; k) = U_real - vhat_k(root).
Per checkpoint k != 74: disagreement rate vs it74's A_act signs;
roots with |A_act| < 1e-6 (in either checkpoint) excluded and
counted. Endgame root = within 50 steps of its source game's end
(dist_to_end <= 50, from realized_u).
  - '指认形态': rate_95 >= 15% AND endgame_rate_95 >= 2x non-endgame
    AND rate_79 >= 50% of rate_95.
  - '证伪形态': rate_95 < 5% OR endgame/non-endgame ratio < 1.25.
  - else '无判定（形态未达任一线）'.

Test C trigger (C itself is NOT run here): triggered iff A verdict
is 指认 (either group label) OR (A is 中间 AND B is 指认形态).
A 证伪 AND B 证伪形态 -> not triggered (value side not implicated).
C caliber noted for the record: seeds 990400-990499 (seat-swap
pairs, 200 games/arm) with the not-constructible rider (if the it74
value head cannot be faithfully transplanted into the search player,
C reports '不可构造' + reason and stops; no improvised substitute).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CKPTS, OUT  # noqa: E402

TINY = 1e-6


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman(x, y):
    return pearson(_rank(x), _rank(y))


def stats(vhat, U, mask):
    e = vhat[mask] - U[mask]
    return {
        "n": int(mask.sum()),
        "bias": float(e.mean()),
        "mae": float(np.abs(e).mean()),
        "pearson": pearson(vhat[mask], U[mask]),
        "spearman": spearman(vhat[mask], U[mask]),
    }


def main():
    ru = np.load(os.path.join(OUT, "realized_u.npz"))
    U = ru["U"]; seat = ru["seat"]; step = ru["step"]; dist = ru["dist_to_end"]
    vh = {}
    missing = []
    for k in CKPTS:
        p = os.path.join(OUT, f"vhat_{k}.npz")
        if os.path.exists(p):
            vh[k] = np.load(p)["vhat"]
        else:
            missing.append(k)
    avail = [k for k in CKPTS if k in vh]

    bands = {"200-249": (step >= 200) & (step <= 249),
             "250-299": (step >= 250) & (step <= 299)}
    A = {}
    for k in avail:
        row = {"overall": stats(vh[k], U, np.ones(len(U), bool))}
        for sd in (0, 1):
            row[f"seat{sd}"] = stats(vh[k], U, seat == sd)
        for bn, bm in bands.items():
            row[f"band{bn}"] = stats(vh[k], U, bm)
            for sd in (0, 1):
                row[f"seat{sd}_band{bn}"] = stats(vh[k], U, bm & (seat == sd))
        A[k] = row

    def metric_seq(which):
        return {k: (A[k]["overall"]["mae"] if which == "mae"
                    else abs(A[k]["overall"]["bias"])) for k in avail}

    a_detail = {}
    labels = set()
    any_fired = False
    for which in ("mae", "bias"):
        m = metric_seq(which)
        delta = m[95] - m[74]
        fired = delta >= 0.020
        any_fired = any_fired or fired
        head_group = (m[78] - m[74]) + (m[79] - m[78])
        r19_steps = None
        if 82 in m:
            r19_steps = [m[82] - m[79], m[85] - m[82]]
            r19_group = r19_steps[0] + r19_steps[1]
        else:
            r19_group = m[85] - m[79]   # telescoped signed sum (exact)
        entry = {"delta": float(delta), "fired": bool(fired),
                 "head_group": float(head_group),
                 "r19_group": float(r19_group),
                 "r19_single_steps": r19_steps,
                 "head_share": None, "r19_share": None, "labels": []}
        if fired and delta > 0:
            entry["head_share"] = float(head_group / delta)
            entry["r19_share"] = float(r19_group / delta)
            if entry["head_share"] >= 0.40:
                entry["labels"].append("指认（换头步）")
            if entry["r19_share"] >= 0.40:
                entry["labels"].append("指认（r19 步）")
            labels.update(entry["labels"])
        a_detail[which] = entry
    d_mae = a_detail["mae"]["delta"]; d_bias = a_detail["bias"]["delta"]
    if d_mae <= 0.005 and d_bias <= 0.005:
        a_verdict = "证伪"
    elif labels:
        a_verdict = "指认：" + "、".join(sorted(labels))
    elif any_fired:
        a_verdict = "中间（退化成立、步归因不成立）"
    else:
        a_verdict = "无判定（量级未达指认门且未达证伪门）"

    # ---- Test B ----
    A74 = U - vh[74]
    endgame = dist <= 50
    B = {}
    for k in avail:
        if k == 74:
            continue
        Ak = U - vh[k]
        ex74 = np.abs(A74) < TINY
        exk = np.abs(Ak) < TINY
        keep = ~(ex74 | exk)
        dis = np.sign(Ak[keep]) != np.sign(A74[keep])

        def rate(mask_full):
            m = mask_full & keep
            if m.sum() == 0:
                return None, 0
            d = np.sign(Ak[m]) != np.sign(A74[m])
            return float(d.mean()), int(m.sum())
        r_all, _ = rate(np.ones(len(U), bool))
        r_end, n_end = rate(endgame)
        r_non, n_non = rate(~endgame)
        ratio = None
        if r_end is not None and r_non is not None:
            if r_non == 0:
                ratio = float("inf") if r_end > 0 else None
            else:
                ratio = r_end / r_non
        B[k] = {"rate": r_all, "rate_endgame": r_end, "rate_nonend": r_non,
                "ratio_end_over_nonend": ratio,
                "n_kept": int(keep.sum()), "n_excluded": int((~keep).sum()),
                "n_excl_a74_tiny": int(ex74.sum()),
                "n_excl_ak_tiny": int(exk.sum()),
                "n_endgame_kept": n_end, "n_nonend_kept": n_non}
    r95 = B[95]["rate"]; ratio95 = B[95]["ratio_end_over_nonend"]
    r79 = B[79]["rate"]
    if (r95 is not None and r95 >= 0.15 and ratio95 is not None
            and ratio95 >= 2.0 and r79 is not None and r79 >= 0.5 * r95):
        b_verdict = "指认形态"
    elif (r95 is not None and r95 < 0.05) or \
            (ratio95 is not None and ratio95 < 1.25):
        b_verdict = "证伪形态"
    else:
        b_verdict = "无判定（形态未达任一线）"

    c_trigger = bool(labels) or (a_verdict.startswith("中间")
                                  and b_verdict == "指认形态")
    if a_verdict == "证伪" and b_verdict == "证伪形态":
        c_trigger = False
        c_note = "A 证伪且 B 证伪形态：C 不触发（价值侧未指认，诊断关闭待终审另议）"
    elif c_trigger:
        c_note = ("C 触发（条件件，另行派工）：种子 990400-990499（席位互换对、"
                  "200 局/臂）；不可构造骑缝——it74 价值头无法保真拼回搜索体则 "
                  "C 报「不可构造」+ 原因并停，不设替代臂")
    else:
        c_note = "C 不触发"

    out = {
        "checkpoints_available": avail,
        "checkpoints_missing": missing,
        "A_table": {str(k): A[k] for k in avail},
        "A_detail": a_detail,
        "A_verdict": a_verdict,
        "B_table": {str(k): B[k] for k in B},
        "B_verdict": b_verdict,
        "C_trigger": c_trigger,
        "C_note": c_note,
    }
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"A_verdict": a_verdict, "B_verdict": b_verdict,
                      "C_trigger": c_trigger,
                      "delta_mae": d_mae, "delta_abs_bias": d_bias,
                      "missing": missing}, ensure_ascii=False, indent=1))
    print("results ->", os.path.join(OUT, "results.json"))


if __name__ == "__main__":
    main()
