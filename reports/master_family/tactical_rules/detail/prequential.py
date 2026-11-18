"""Compare previous-turn-only adaptation and fixed predictions on other Top 5 games."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from tree_adaptation import AdaptiveTreeIntent

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from opponent_intent import predict
from top5_intent_check import own_view


def choose(scores, ranks, excluded):
    """Score-blind order, 4 in each actual seat per player, no Top 5 duels."""
    selected = []
    for name in ranks:
        groups = [[], []]
        for key, game in scores.items():
            names = [game["name_" + side] for side in game["order"]]
            if key in excluded or name not in names or names[1 - names.index(name)] in ranks:
                continue
            order = hashlib.sha256(("sapling-tree-style-v1:" + key).encode()).digest()
            seat = names.index(name)
            groups[seat].append((order, key))
        if min(map(len, groups)) < 4:
            raise ValueError("Insufficient games")
        for seat, group in enumerate(groups):
            selected.extend((key, name, seat) for _, key in sorted(group)[:4])
    return selected


def measure(task):
    key, name, seat, path, expected, score, weights, reference = task
    raw = path.read_bytes()
    replay = json.loads(gzip.decompress(raw))
    if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        canonical = json.dumps(replay, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        if hashlib.sha256(canonical).hexdigest() != expected["canonical_sha256"]:
            raise ValueError(f"Replay identity {key}")
    if replay["rewards"] != [score["scores"][s] for s in score["order"]]:
        raise ValueError(f"Seat issue: {key}")
    memory = AdaptiveTreeIntent()
    previous = None
    rows = []
    snapshots = replay["steps"]
    for t, (current, following) in enumerate(zip(snapshots, snapshots[1:])):
        obs = own_view(current, 1 - seat)
        cfg = replay["configuration"]
        memory.observe(obs, cfg)
        if not reference and memory.pending:
            frozen = predict(obs, weights, previous)
            live = memory.probabilities(obs, cfg, frozen)
            # Only the evaluator receives enemy submissions. Disappearing units are scored too.
            actions = following[seat]["action"] or {}
            for uid, (_, context) in memory.pending.items():
                y = int((actions.get(uid) or "NONE") == "NONE")
                evidence = float(memory.counts[int(context), 1])
                rows.append((t, context, evidence, y, float(frozen[uid][0]), float(live[uid][0])))
        elif reference and memory.pending:
            # Reference cohort calibrates a fixed baseline; use only observable survivor moves.
            next_enemies = following[0]["observation"]["players"][seat][2]
            for uid, (pos, context) in memory.pending.items():
                if uid in next_enemies:
                    rows.append((context, int(next_enemies[uid][0] == pos)))
        previous = obs
    return key, name, seat, rows, memory.learned, memory.censored


def report_rows(rows):
    if not rows:
        return {"observations": 0}
    a = np.asarray(rows)
    y = a[:, 0]
    out = {"observations": len(y), "actual_stays": int(y.sum())}
    for col, model in enumerate(("frozen", "online", "global_same_weight", "global_context", "always_half"), 1):
        probability = a[:, col]
        safe = np.clip(probability, 1e-9, 1 - 1e-9)
        out[model] = {"mean_forecast": float(probability.mean()),
                      "brier": float(np.square(probability - y).mean()),
                      "nll": float(-(y*np.log(safe)+(1-y)*np.log1p(-safe)).mean())}
    return out


def evaluate(results, reference, prior_weight):
    groups = {"all": [], "covered": [], "uncovered": [], "first_observations": [], "history_ge_3": []}
    games = []
    for key, name, seat, rows, learned, censored in results:
        local = []
        for t, covered, n, y, frozen, online in rows:
            generic = (prior_weight * frozen + n * reference[int(covered)]) / (prior_weight + n)
            values = (y, frozen, online, generic, reference[int(covered)], .5)
            local.append(values)
            groups["all"].append(values)
            groups["covered" if covered else "uncovered"].append(values)
            if n == 0:
                groups["first_observations"].append(values)
            if n >= 3:
                groups["history_ge_3"].append(values)
            groups.setdefault(name, []).append(values)
        games.append({"key": key, "model": name, "seat": seat, "resolved_through_final_decision": learned,
                      "censored_through_final_decision": censored, **report_rows(local)})
    rng = np.random.default_rng(1105)
    pair = {}
    for opponent in ("frozen", "global_same_weight", "global_context"):
        by_rank = []
        for player in sorted({g["model"] for g in games}):
            group = [g["online"]["brier"] - g[opponent]["brier"] for g in games
                     if g["model"] == player and g["observations"]]
            if not group:
                raise ValueError("There must be eligible games for each rank")
            by_rank.append(np.asarray(group))
        draws = [rng.choice(group, (4000, len(group)), replace=True).mean(axis=1) for group in by_rank]
        difference = np.asarray([group.mean() for group in by_rank]).mean()
        pair[opponent] = {"rank_then_game_equal_weight_mean_brier_delta_online_minus_comparator": float(difference),
                          "improved_games": sum(int((g < 0).sum()) for g in by_rank),
                          "eligible_games": sum(map(len, by_rank)),
                          "ci95_stratified_game_bootstrap": np.quantile(np.mean(draws, axis=0), [.025,.975]).tolist()}
    return {"groups": {k: report_rows(v) for k, v in groups.items()}, "games": games, "paired": pair}


def main():
    from tree_adaptation import HALF_LIFE, PRIOR_WEIGHT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replays", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    weights = np.asarray(json.loads((ROOT / "opponent_intent.json").read_bytes())["weights"])
    index = json.loads((ROOT / "top5_inventory.json").read_bytes())
    score_bytes = (ROOT / "top5_score_rows.json").read_bytes()
    if hashlib.sha256(score_bytes).hexdigest() != index["score_rows_sha256"]:
        raise ValueError("Score sheet mismatch")
    scores = json.loads(score_bytes)
    prior_test = json.loads((ROOT / "top5_intent_check.json").read_bytes())["cohorts"]
    earlier = [(g["key"], model, g["seat"]) for model, data in prior_test.items() for g in data["games"]]
    selected = choose(scores, tuple(prior_test), {key for key, _, _ in earlier})
    if len({k for k, _, _ in earlier + selected}) != 80:
        raise ValueError("Overlapping games")
    tasks = [(key, model, seat, args.replays / (key + ".json.gz"), index["replays"][key],
              scores[key], weights, i < len(earlier)) for i, (key, model, seat) in enumerate(earlier + selected)]
    with ProcessPoolExecutor(2) as pool:
        result = list(pool.map(measure, tasks, chunksize=3))
    counts = [Counter(), Counter()]
    for _, _, _, rows, _, _ in result[:len(earlier)]:
        for context, label in rows:
            counts[int(context)][label] += 1
    rate = [(c[1] + 1) / (c.total() + 2) for c in counts]
    report = {"score_sha256": index["score_rows_sha256"],
              "sources_sha256": {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(__file__).with_name("tree_adaptation.py"), ROOT / "opponent_intent.json")}, "prior_weight": PRIOR_WEIGHT,
              "decay_half_life_decisions": HALF_LIFE, "reference_games": [k for k,_,_ in earlier],
              "historical_survivor_counts_by_uncovered_covered": counts, "reference_rates": rate,
              "protocol": "40 reference, disjoint 40 diagnostic games; 4 per seat per player. Fresh state each match. "
                          "No enemy submissions supplied to live module. Score every eligible command, including "
                          "workers that die. Global baseline uses other 40 games with identical effective evidence "
                          "weight by context. Observation contexts shared across all enemy planters in a game.",
              **evaluate(result[len(earlier):], rate, PRIOR_WEIGHT)}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
