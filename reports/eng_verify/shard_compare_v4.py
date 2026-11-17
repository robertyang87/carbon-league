#!/usr/bin/env python3
"""Per-(root, action) shard identity comparator — shard_compare v4.

v4 = v3 (17b4ec07) with ONLY the join key changed, per strategy-group
ruling reports/RULING_SMALL_TEACHER_COMPARE_ALIGNMENT_20261118.md
(commit ce48e03e, md5 9b2f6130):
  - v3 joined per root on arm identity but required the two files' arm
    SETS to be equal; roots with differing sets were excluded from all
    q statistics (994400: compared_roots 0; 994401: compared_roots 1).
  - v4 joins on (game, step, action): same root (step, seat in the same
    game file), same action (the 12-int arm encoding). Only actions
    present in BOTH files for a root are compared; roots with no common
    action contribute nothing and are counted separately.
  - Tier 1 / Tier 2 four-item definitions and thresholds are VERBATIM
    unchanged (only the join key changed). All q / flip / du statistics
    are computed over the compared (root, action) entries exactly as v3
    computed them over compared roots' full arm sets.
  - Old v3 values for 994401 (single root: q_max_abs_delta 0.0907,
    argmax_flips 1) are VOID per the ruling and must not be cited.
  - Comparability floor (ruling section 2): compared_roots < 100 over
    the 396-root appendix window => Tier 1/2 criteria are not
    statistically meaningful; no hard Tier judgement may be made. The
    report carries compared_roots and comparable_action_entries per game
    and in total; Tier judgement belongs to the strategy group.

Statistics (per game; strategy-group pre-registered yardstick, 2026-11-16):
  1. q |dq| distribution: max over compared (root, action) entries, plus
     quantiles (p50/p90/p99) of the per-root delta (max-arm |dq| per root).
  2. signed mean bias (MAIN criterion): mean of (q_cand - q_base) over
     all compared (root, action) entries, per game.
  3. argmax flips: count + rate over compared roots (argmax by arm
     identity over the COMMON action set; ties -> lowest slot in each
     file's own order); every flipped root's delta attached; flips split
     small (|dq| <= 1e-3) vs large.
  4. du: du_root := max_a q_a - sum_a p~_a * q_a, p~ = priors renormalized
     over the COMMON action set (shard-native decision-value proxy).
     Report the per-root |ddu| distribution (p50/p90/p99/max) and the
     per-game mean signed ddu (drift).
  Reference noise floors for judgement: sigma_root ~ 0.209,
  sigma_game ~ 0.021 (S2-1 measured); drifts/bias are judged far below
  sigma_game.
  Also kept: frac_roots_delta_pos (exact >0 and at 1e-9/1e-6/1e-3).

Identity pre-checks: candidate _meta identity config_sha256 / games_sha256 /
per-game replay_sha256 must equal baseline's; code_md5s are reported but
expected to differ for candidate code.
"""
import json
import sys

import numpy as np

TOLS = (1e-9, 1e-6, 1e-3)

# Tier thresholds (verbatim from the small-teacher registration; unchanged
# by the alignment ruling — only the join key changed).
TIER1 = {"argmax_flips": 0, "abs_signed_mean_bias": 1e-4,
         "q_max_abs_delta": 1e-3, "du_max_abs_delta": 1e-3}
TIER2 = {"argmax_flip_rate": 0.02, "abs_signed_mean_bias": 1e-3,
         "q_max_abs_delta": 5e-2, "du_max_abs_delta": 5e-2}
COMPARABILITY_FLOOR = 100  # ruling section 2, over the 396-root window


def load(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["_meta_parsed"] = json.loads(str(z["_meta"]))
    return d


def arm_key(row):
    return tuple(int(x) for x in row)


def compare_game(base, cand):
    out = {"structural_mismatch": None}
    bkeys = [(int(s), int(t)) for s, t in zip(base["step"], base["seat"])]
    ckeys = [(int(s), int(t)) for s, t in zip(cand["step"], cand["seat"])]
    if len(set(bkeys)) != len(bkeys) or len(set(ckeys)) != len(ckeys):
        out["structural_mismatch"] = "duplicate (step, seat) keys"
        return out
    bidx = {k: i for i, k in enumerate(bkeys)}
    pairs = []  # (base_row, cand_row)
    for j, k in enumerate(ckeys):
        if k not in bidx:
            out["structural_mismatch"] = f"candidate root {k} not in baseline"
            return out
        pairs.append((bidx[k], j))
    n = len(pairs)
    out["n_roots"] = int(n)
    out["baseline_roots"] = int(len(bkeys))
    # identity pre-checks
    bi, ci = base["_meta_parsed"]["identity"], cand["_meta_parsed"]["identity"]
    out["identity_match"] = {
        k: (bi.get(k) == ci.get(k)) for k in ("config_sha256", "games_sha256")
    }
    game = base["_meta_parsed"]["game"]
    out["replay_sha_match"] = (
        bi["replay_sha256"].get(game) == ci["replay_sha256"].get(game)
    )
    max_dq = 0.0
    root_delta = np.zeros(n)
    compared_mask = np.zeros(n, dtype=bool)
    flips = 0
    flip_deltas = []
    arms_set_diff = 0
    roots_no_common_action = 0
    roots_partial_overlap = 0
    roots_exact_arm_set = 0
    du_base = np.zeros(n)
    du_cand = np.zeros(n)
    signed_dq_entries = []
    compared = 0
    for r, (br, cr) in enumerate(pairs):
        nb = int(base["n_arms"][br])
        nc = int(cand["n_arms"][cr])
        bmap = {arm_key(base["arms"][br, k]): k for k in range(nb)}
        cmap = {arm_key(cand["arms"][cr, k]): k for k in range(nc)}
        if set(bmap) != set(cmap):
            arms_set_diff += 1
        else:
            roots_exact_arm_set += 1
        # v4 join: (game, step, action) — common actions only.
        keys = [k for k in bmap if k in cmap]
        if not keys:
            roots_no_common_action += 1
            continue
        if len(keys) < max(nb, nc):
            roots_partial_overlap += 1
        compared += 1
        compared_mask[r] = True
        qb = np.array([base["q"][br, bmap[k]] for k in keys])
        qc = np.array([cand["q"][cr, cmap[k]] for k in keys])
        signed_dq_entries.extend((qc - qb).tolist())
        dq = np.abs(qc - qb)
        root_delta[r] = dq.max() if len(dq) else 0.0
        max_dq = max(max_dq, float(root_delta[r]))
        if keys[int(np.argmax(qb))] != keys[int(np.argmax(qc))]:
            flips += 1
            flip_deltas.append(float(root_delta[r]))
        pb = np.array([base["priors"][br, bmap[k]] for k in keys], dtype=np.float64)
        pc = np.array([cand["priors"][cr, cmap[k]] for k in keys], dtype=np.float64)
        du_base[r] = qb.max() - float((pb / pb.sum()) @ qb) if pb.sum() > 0 else qb.max()
        du_cand[r] = qc.max() - float((pc / pc.sum()) @ qc) if pc.sum() > 0 else qc.max()
    out["arms_set_diff_roots"] = int(arms_set_diff)
    out["roots_exact_arm_set"] = int(roots_exact_arm_set)
    out["roots_partial_overlap"] = int(roots_partial_overlap)
    out["roots_no_common_action"] = int(roots_no_common_action)
    out["compared_roots"] = int(compared)
    out["comparable_action_entries"] = int(len(signed_dq_entries))
    # fractions are over compared roots only; no-common-action roots are
    # reported separately above and enter no q statistic.
    rd = root_delta[compared_mask]

    def quantiles(x):
        if len(x) == 0:
            return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
        return {
            "p50": float(np.quantile(x, 0.50)),
            "p90": float(np.quantile(x, 0.90)),
            "p99": float(np.quantile(x, 0.99)),
            "max": float(np.max(x)),
        }

    # 1. q |dq| distribution (max = max over compared (root, action) entries)
    out["q_abs_delta_root_quantiles"] = quantiles(rd)
    out["q_max_abs_delta"] = float(max_dq)
    # 2. signed mean bias (main criterion), over (root, action) entries
    sde = np.array(signed_dq_entries)
    out["q_signed_mean_bias"] = float(sde.mean()) if len(sde) else 0.0
    out["q_signed_entries"] = int(len(sde))
    out["frac_roots_delta_pos"] = float(np.mean(rd > 0)) if compared else 0.0
    out["frac_roots_delta_gt_tol"] = {
        f"{t:g}": float(np.mean(rd > t)) if compared else 0.0 for t in TOLS
    }
    # 3. flips: count, rate, per-flip |dq|, small/large split at 1e-3
    out["argmax_flips"] = int(flips)
    out["argmax_flip_rate"] = float(flips / compared) if compared else 0.0
    out["flip_root_deltas"] = flip_deltas
    out["flips_small_le_1e-3"] = int(sum(1 for d in flip_deltas if d <= 1e-3))
    out["flips_large_gt_1e-3"] = int(sum(1 for d in flip_deltas if d > 1e-3))
    # 4. du: per-root |ddu| distribution + per-game mean signed drift
    ddu = (du_cand - du_base)[compared_mask]
    out["du_abs_delta_root_quantiles"] = quantiles(np.abs(ddu))
    out["du_max_abs_delta"] = float(np.abs(ddu).max()) if compared else 0.0
    out["du_mean_signed_delta"] = float(ddu.mean()) if compared else 0.0
    # Tier 1/2 four-item readings (definitions and thresholds verbatim
    # from the registration; judgement belongs to the strategy group).
    out["tier1"] = {
        "argmax_flips": int(flips),
        "abs_signed_mean_bias": float(abs(out["q_signed_mean_bias"])),
        "q_max_abs_delta": float(max_dq),
        "du_max_abs_delta": out["du_max_abs_delta"],
    }
    out["tier2"] = {
        "argmax_flip_rate": out["argmax_flip_rate"],
        "abs_signed_mean_bias": float(abs(out["q_signed_mean_bias"])),
        "q_max_abs_delta": float(max_dq),
        "du_max_abs_delta": out["du_max_abs_delta"],
    }
    return out


def main(base_paths, cand_paths):
    report = {}
    for bp, cp in zip(base_paths, cand_paths):
        base, cand = load(bp), load(cp)
        game = base["_meta_parsed"]["game"]
        report[game] = compare_game(base, cand)
    games = [g for g in report if g != "_total"]
    tot_compared = sum(report[g]["compared_roots"] for g in games)
    tot_entries = sum(report[g]["comparable_action_entries"] for g in games)
    tot_flips = sum(report[g]["argmax_flips"] for g in games)
    tot_signed = sum(report[g]["q_signed_mean_bias"] * report[g]["q_signed_entries"]
                     for g in games)
    report["_total"] = {
        "n_roots": sum(report[g]["n_roots"] for g in games),
        "compared_roots": int(tot_compared),
        "comparable_action_entries": int(tot_entries),
        "argmax_flips": int(tot_flips),
        "argmax_flip_rate": float(tot_flips / tot_compared) if tot_compared else 0.0,
        "q_signed_mean_bias": float(tot_signed / tot_entries) if tot_entries else 0.0,
        "q_max_abs_delta": max((report[g]["q_max_abs_delta"] for g in games), default=0.0),
        "du_max_abs_delta": max((report[g]["du_max_abs_delta"] for g in games), default=0.0),
        "comparability_floor": COMPARABILITY_FLOOR,
        "floor_triggered": bool(tot_compared < COMPARABILITY_FLOOR),
        "tier1_thresholds": TIER1,
        "tier2_thresholds": TIER2,
    }
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    # usage: shard_compare_v4.py base1.npz cand1.npz [base2.npz cand2.npz]
    args = sys.argv[1:]
    assert len(args) in (2, 4), "expected 2 or 4 npz paths"
    main(args[0::2], args[1::2])
