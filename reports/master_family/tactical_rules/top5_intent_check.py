"""Measure a frozen enemy-move predictor on real, unseen players from the heavy ladder.

Select four archived games in each seat per model by a score-blind hash order.
Read only the viewing player's masked carbon. Keep rows for enemies killed by
the transition. The unit of replication is the game. No fitting on these games.
"""
import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

import opponent_intent as intent
import policy_v0_5 as heuristic

HERE = Path(__file__).resolve().parent
REV = "88b4b108"
COHORTS = {
    "master5s": "claude-AlphaCarbon-Master5s",
    "master6": "claude-AlphaCarbon-Master6",
    "it720": "RyanX-AIPowercore-it720-preview",
    "m9xj630": "RyanX-AIPowercore-M9XJ-630",
    "fablev7": "ClaudeFable5.2-v7",
}


def git(*args):
    return subprocess.check_output(["git", "-C", str(intent.ROOT), *args])


def choose_games(corpus, model):
    metadata = json.loads(git("show", f"{REV}:data/{corpus}_matches.json"))
    paths = set(git("ls-tree", "--name-only", f"{REV}:data/{corpus}_corpus").decode().splitlines())
    candidates = [[], []]
    for game in metadata:
        names = [game["name_" + side] for side in game["order"]]
        if names.count(model) != 1 or names[1 - names.index(model)] in COHORTS.values():
            continue
        key = f"{game['id']}-{game['game']}"
        if key + ".json.gz" in paths:
            rank = hashlib.sha256(("opponent-intent-v1/" + key).encode()).digest()
            candidates[names.index(model)].append((rank, key, game))
    if min(map(len, candidates)) < 4:
        raise ValueError(f"Insufficient separate seats in {corpus}")
    return [(seat, key, game) for seat, group in enumerate(candidates)
            for _, key, game in sorted(group, key=lambda row: row[0])[:4]]


def own_view(frame, seat):
    shared = frame[0]["observation"]
    own = frame[seat]["observation"]
    if own["player"] != seat:
        raise ValueError("Replay seat mismatch")
    return {"player": seat, "step": shared["step"], "players": shared["players"],
            "trees": shared["trees"], "carbon": own["carbon"]}


def score_game(corpus, seat, key, game, weights, frequency):
    path = f"data/{corpus}_corpus/{key}.json.gz"
    raw = git("show", f"{REV}:{path}")
    replay = json.loads(gzip.decompress(raw))
    if replay["rewards"] != [game["scores"][s] for s in game["order"]]:
        raise ValueError(f"Score/seat mismatch in {key}")
    steps = replay["steps"]
    belief = heuristic.BeliefState()
    predictor = heuristic.OpponentModel(belief)
    previous = None
    rows = []
    for t, (frame, following) in enumerate(zip(steps, steps[1:])):
        obs = own_view(frame, 1 - seat)
        if t != obs["step"]:
            raise ValueError(f"Missing step in {key}")
        belief.update(obs)
        if t % 5 == 0:
            ids, roles, features = intent.features(obs, previous)
            board = heuristic.BoardView(obs)
            plant_value = heuristic.compute_plant_value(belief.belief, board)
            baseline = predictor.predict(board, plant_value)
            # Reveal logged actions only after constructing all predictor inputs.
            workers = obs["players"][1-seat][2]
            positions = [w[0] for w in workers.values()]
            choices = intent.recorded_moves(following[1-seat]["action"] or {}, workers)
            arrivals = {int(intent.DEST[p, a]) for p, a in zip(positions, choices)}
            labels = intent.recorded_moves(following[seat]["action"] or {}, ids)
            for j, uid in enumerate(ids):
                location = obs["players"][seat][2][uid][0]
                cells = intent.DEST[location]
                reachable = bool(positions and intent.DIST[location, positions].min() <= 2)
                meets = [int(cell) in arrivals for cell in cells]
                scores = dict(baseline[uid])
                rows.append((key, roles[j], features[j], labels[j], reachable, meets,
                             [scores.get(int(c), 0.0) for c in cells]))
        previous = obs
    data = tuple(np.asarray(column) for column in zip(*rows))
    summary = intent.evaluate(data, weights, frequency)
    return data, {"key": key, "seat": seat, "source": path, "sha256": hashlib.sha256(raw).hexdigest(),
                  "opponent": game["name_" + game["order"][1-seat]], "turns": len(steps) - 1,
                  "within_two_cells": summary["within_two_cells"]}


def aggregate(batches, weights, frequency):
    combined = tuple(np.concatenate(column) for column in zip(*batches))
    stats = intent.evaluate(combined, weights, frequency)
    return {key: stats[key] for key in ("all", "within_two_cells")}


def paired_by_game(cohorts):
    """Equal weight per game; resample games within each named player."""
    delta = []
    for cohort in cohorts.values():
        scores = [game["within_two_cells"] for game in cohort["games"]]
        delta.append([s["learned"]["meeting_brier"] - s["existing_rule"]["meeting_brier"]
                      for s in scores])
    delta = np.asarray(delta)
    players, games = delta.shape
    rng = np.random.default_rng(2818)
    indices = rng.integers(games, size=(4000, players, games))
    samples = delta[np.arange(players)[None, :, None], indices].mean(axis=(1, 2))
    return {"metric": "near-worker meeting Brier, learned minus existing rule; equal games; "
                      "95% percentile interval conditional on these five players",
            "games": int(delta.size), "games_improved": int((delta < 0).sum()),
            "mean_delta": float(delta.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=HERE / "opponent_intent.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frozen = args.weights.read_bytes()
    trained = json.loads(frozen)
    assert trained["features"] == intent.FEATURES and trained["action_order"] == intent.MOVES
    weights, frequency = np.asarray(trained["weights"]), np.asarray(trained["role_frequency"])
    result = {"revision": git("rev-parse", REV).decode().strip(),
              "frozen_model_receipt_sha256": hashlib.sha256(frozen).hexdigest(),
              "protocol": "8 games/player: 4 per actual seat, smallest salted hashes among games archived at revision, opponent outside five. Every fifth pre-action state over the actual game. Both sides' submitted moves read after prediction. Actual meeting = co-destination with an existing opponent, no recruit. No parameter tuning.",
              "cohorts": {}}
    batches, seen = [], set()
    for corpus, model in COHORTS.items():
        group, records = [], []
        for seat, key, game in choose_games(corpus, model):
            if key in seen:
                raise ValueError(f"Duplicate match {key}")
            seen.add(key)
            data, record = score_game(corpus, seat, key, game, weights, frequency)
            group.append(data)
            records.append(record)
        result["cohorts"][model] = {"games": records, **aggregate(group, weights, frequency)}
        batches.extend(group)
    result["combined"] = aggregate(batches, weights, frequency)
    result["paired_by_game"] = paired_by_game(result["cohorts"])
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
