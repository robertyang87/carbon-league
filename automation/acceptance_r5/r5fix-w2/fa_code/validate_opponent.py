"""Offline validation of policy_v1's OpponentModel on public replays.

For each replay game and each player perspective, the predictor is fed
that player's own observation stream (belief state built incrementally,
exactly as in live play), predicts every enemy unit's next cell, and the
top-1 prediction is compared with the unit's actual position one step
later (taken from the shared players table). Units that die that step
are counted separately and excluded from accuracy.

Baselines: uniform random over 5 options = 20%; "always stay" predictor.

Run: python3 validate_opponent.py [max_games]
Writes opponent_validation.json with the aggregates.
"""

import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import policy_v1 as PV  # noqa: E402


def reconstruct_obs(steps, t, player):
    """Player `player`'s observation at step t of a replay. steps[t][0]
    carries the shared fields (players / trees / step); steps[t][1] was
    stripped to carbon/player/remainingOverageTime (see bc_data README)."""
    base = steps[t][0]["observation"]
    if player == 0:
        return base
    own = steps[t][player]["observation"]
    return {
        "carbon": own["carbon"],
        "players": base["players"],
        "player": player,
        "step": base["step"],
        "trees": base["trees"],
        "remainingOverageTime": own.get("remainingOverageTime", 5),
    }


def classify(typ, cargo):
    if typ == "PLANTER":
        return "planter"
    if cargo >= 150:
        return "collector_returning"
    return "collector_harvest"


def main():
    max_games = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    index = json.load(open(os.path.join(HERE, "replays", "index.json")))
    games = [g for g in index if g.get("scores") and g["scores"].get("a") is not None]
    games = games[:max_games]
    stats = {}
    died = 0
    total = 0

    def bump(cls, hit, stay_hit):
        s = stats.setdefault(cls, {"n": 0, "top1": 0, "stay": 0})
        s["n"] += 1
        s["top1"] += 1 if hit else 0
        s["stay"] += 1 if stay_hit else 0

    for gi, g in enumerate(games):
        path = os.path.join(HERE, "replays", g["file"])
        with gzip.open(path, "rt") as f:
            replay = json.load(f)
        steps = replay["steps"]
        for player in range(2):
            belief = PV.BeliefState()
            model = PV.OpponentModel(belief)
            for t in range(len(steps) - 1):
                obs = reconstruct_obs(steps, t, player)
                belief.update(obs)
                view = PV.BoardView(obs)
                if not view.opp_workers:
                    continue
                pv_map = PV.compute_plant_value(belief.belief, view)
                preds = model.predict(view, pv_map)
                truth = steps[t + 1][0]["observation"]["players"][1 - player][2]
                for wid, (pos, cargo, typ) in view.opp_workers.items():
                    if wid not in preds:
                        continue
                    if wid not in truth:
                        died += 1
                        continue
                    actual = truth[wid][0]
                    top1 = preds[wid][0][0]
                    cls = classify(typ, cargo)
                    bump(cls, top1 == actual, pos == actual)
                    bump("ALL", top1 == actual, pos == actual)
                    total += 1
        if (gi + 1) % 20 == 0:
            print(f"{gi + 1} games done, samples={total}", flush=True)

    result = {"games": len(games), "samples": total, "died_excluded": died,
              "random_baseline": 0.20, "stats": {}}
    for cls, s in sorted(stats.items()):
        result["stats"][cls] = {
            "n": s["n"],
            "top1_acc": round(s["top1"] / s["n"], 4) if s["n"] else None,
            "stay_baseline_acc": round(s["stay"] / s["n"], 4) if s["n"] else None,
        }
    with open(os.path.join(HERE, "opponent_validation.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
