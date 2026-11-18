"""Confirm the fixed two-context tree-defense prior with and without current-match evidence.

Only the observer's earlier visible survivor movements update the estimator. The
reference prior stays fixed; the evaluator scores all submitted planter actions.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from prequential import ROOT, choose, measure, own_view
from tree_adaptation import AdaptiveTreeIntent, HALF_LIFE, PRIOR_WEIGHT

HERE = Path(__file__).resolve().parent
NAMES = ("frozen_model", "frozen_model_plus_history", "fixed_context", "context_plus_history")


def metrics(data):
    if not data:
        return {"opportunities": 0, "stays": 0, "games": 0}
    y = np.array([r[1] for r in data])
    out = {"opportunities": len(data), "stays": int(y.sum()), "games": len({r[0] for r in data})}
    for i, name in enumerate(NAMES, 2):
        p = np.array([r[i] for r in data])
        safe = np.clip(p, 1e-9, 1 - 1e-9)
        out[name] = {"brier": float(np.mean((p - y)**2)),
                     "log_loss": float(np.mean(-y*np.log(safe) - (1-y)*np.log1p(-safe))),
                     "mean_probability": float(p.mean())}
    return out


def forecast_rows(game, prior):
    key, name, seat, observations, learned, censored = game
    groups = {tag: [] for tag in ("all", "covered", "uncovered", "no_history", "any_history", "effective_n_ge3")}
    for _, covered, n, y, frozen, online in observations:
        # Invert the pre-existing unrounded tracker formula to recover only its
        # observable discounted success count. No opposite-seat action is used
        # in this calculation; y is held only for scoring the predictions.
        recent_stays = (PRIOR_WEIGHT + n) * online - PRIOR_WEIGHT * frozen
        if recent_stays < -1e-9 or recent_stays > n + 1e-9:
            raise ValueError("Invalid discounted observations")
        fixed = prior[int(covered)]
        adjusted = (PRIOR_WEIGHT * fixed + recent_stays) / (PRIOR_WEIGHT + n)
        row = (key, y, frozen, online, fixed, adjusted)
        groups["all"].append(row)
        groups["covered" if covered else "uncovered"].append(row)
        groups["any_history" if n else "no_history"].append(row)
        if n >= 3:
            groups["effective_n_ge3"].append(row)
    return groups


def paired(games, strata, comparator):
    groups = []
    for model in strata:
        values = [g["all"]["context_plus_history"]["brier"] - g["all"][comparator]["brier"]
                  for g in games if g["model"] == model and g["all"]["opportunities"]]
        if not values:
            raise ValueError("One rank has no eligible games")
        groups.append(np.asarray(values))
    rng = np.random.default_rng(1105)
    draws = np.stack([rng.choice(v, size=(4000, len(v))).mean(axis=1) for v in groups])
    return {"rank_then_game_equal_mean_brier_difference": float(np.mean([v.mean() for v in groups])),
            "games_lower_brier": sum(int((v < 0).sum()) for v in groups),
            "games_equal_brier": sum(int((v == 0).sum()) for v in groups),
            "eligible_games": sum(map(len, groups)),
            "ci95_rank_stratified_game_bootstrap": np.quantile(draws.mean(axis=0), [.025,.975]).tolist()}


def verify_observable_counts(path, result):
    """Recheck one whole game's inferred history against the tracker's live counters."""
    replay = json.loads(gzip.decompress(path.read_bytes()))
    _, _, seat, rows, _, _ = result
    tracker = AdaptiveTreeIntent()
    i = 0
    for frame in replay["steps"][:-1]:
        obs = own_view(frame, 1-seat)
        tracker.observe(obs, replay["configuration"])
        for _, (_, context) in tracker.pending.items():
            t, covered, n, _, frozen, online = rows[i]
            recent, total = tracker.counts[int(covered)]
            np.testing.assert_allclose(recent, (PRIOR_WEIGHT+n)*online - PRIOR_WEIGHT*frozen, atol=1e-12, rtol=1e-12)
            if covered != context or t != obs["step"] or abs(n-total)>1e-12:
                raise ValueError("Live context mismatch")
            i += 1
    if i != len(rows):
        raise ValueError("Observation count mismatch")
    return i


def run(folder):
    receipt = json.loads((HERE/"prequential.json").read_bytes())
    inventory = json.loads((ROOT/"top5_inventory.json").read_bytes())
    raw_scores = (ROOT/"top5_score_rows.json").read_bytes()
    if hashlib.sha256(raw_scores).hexdigest() != inventory["score_rows_sha256"]:
        raise ValueError("score table changed")
    scores = json.loads(raw_scores)
    ranks = [r["name"] for r in inventory["ranks"]]
    rates = receipt["reference_rates"]
    reference = set(receipt["reference_games"])
    prior_test = {g["key"] for g in receipt["games"]}
    if len(reference | prior_test) != 80 or len(reference) != len(prior_test):
        raise ValueError("expected two prior 40-game cohorts")
    if receipt["prior_weight"] != PRIOR_WEIGHT or receipt["decay_half_life_decisions"] != HALF_LIFE:
        raise ValueError("Inherited fixed settings changed")
    selected = choose(scores, ranks, reference | prior_test)
    if len({key for key, _, _ in selected} | reference | prior_test) != 120:
        raise ValueError("Independent cohort validation failed")
    weights = np.array(json.loads((ROOT/"opponent_intent.json").read_bytes())["weights"])
    tasks = [(key, name, seat, folder/(key+".json.gz"), inventory["replays"][key],
              scores[key], weights, False) for key, name, seat in selected]
    with ProcessPoolExecutor(2) as pool:
        results = list(pool.map(measure, tasks, chunksize=2))
    union = {}
    game_rows = []
    for result in results:
        key, model, seat, raw_rows, learned, censored = result
        groups = forecast_rows(result, rates)
        for tag, values in groups.items():
            union.setdefault(tag, []).extend(values)
        game_rows.append({"key": key, "model": model, "seat": seat,
                          "updates_seen_through_last_decision": learned,
                          "updates_censored_through_last_decision": censored,
                          **{tag: metrics(values) for tag, values in groups.items()}})
    checks = sum(verify_observable_counts(task[3], result) for task, result in zip(tasks, results))
    return {
        "cohort": "Third disjoint 40-game cohort from the existing 1,056-game diagnostic snapshot. Four games per actual seat per ranked model, no top-five duels. Unused by the first two intent comparisons; all were eligible for prior corpus-level descriptive analysis.",
        "method": "Score affordable stationary-or-moving planter on our live tree, covered or uncovered at the pre-action state. Six prior equivalent observations; for each prior tick multiply earlier counts by 2^(-1/40). State shared over enemy planters by context within one match. Update only when opponent worker ID is visible on the next observation. Evaluate actual submitted actions from all eligible workers, including those later lost. A match's last decision gets scored but its update is not consumed by another forecast.",
        "interpretation": "Fixed context and hybrid use the same rates estimated by survivor movements from the original 40 reference games. No retuning on either diagnostic cohort. Censoring may depend on stay/collision and cannot be repaired from disappeared unit IDs. These probability scores do not estimate match strength or the intent behind an action. Intervals resample matches within the five fixed players; opportunities from one game are correlated.",
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          (Path(__file__), HERE/"prequential.py", HERE/"tree_adaptation.py",
                           HERE/"prequential.json", ROOT/"opponent_intent.json", ROOT/"top5_score_rows.json")},
        "prior_weight": PRIOR_WEIGHT, "half_life_ticks": HALF_LIFE,
        "reference_survivor_counts": receipt["historical_survivor_counts_by_uncovered_covered"],
        "reference_probability_uncovered_covered": rates,
        "excluded_prior_games": sorted(reference | prior_test), "selected_games": len(selected),
        "verified_direct_live_success_counts": checks,
        "groups": {tag: metrics(values) for tag,values in union.items()},
        "paired_hybrid_minus": {name: paired(game_rows, ranks, name) for name in NAMES[:-1]},
        "by_game": game_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(run(args.replays), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
