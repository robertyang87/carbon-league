"""Reconcile the four original-opponent development arms on the registered map."""
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean, median


HERE = Path(__file__).resolve().parent
ARMS = ("parent", "candidate", "hybrid_parent", "hybrid_candidate")


def read_json(path):
    return json.loads(path.read_bytes())


def summarize(name, seed, terrain):
    directory = HERE / name
    result = read_json(directory / "results.json")
    proof = read_json(directory / "official_replay.json")["runs"][name]["games"]
    with gzip.open(directory / "income.json.gz", "rt") as source:
        economy = json.load(source)["games"]
    played = sorted(result["games"], key=lambda r: r["candidate_seat"])
    if (result["status"] != "complete" or [r["candidate_seat"] for r in played] != [0, 1]
            or [r["seat"] for r in proof] != [0, 1] or [r["seat"] for r in economy] != [0, 1]):
        raise ValueError(f"incomplete arm {name}")
    seats = []
    for row, official, cash in zip(played, proof, economy):
        seat = row["candidate_seat"]
        digest = hashlib.sha256((directory / row["raw_file"]).read_bytes()).hexdigest()
        score = official["official_reward_by_seat"]
        margin = round(score[seat] - score[1 - seat], 2)
        expected = "win" if margin > 0 else "draw" if margin == 0 else "loss"
        if not (row["valid"] and not row["errors"] and row["agent_errors_by_seat"] == [0, 0]
                and row["seed"] == official["seed"] == cash["seed"] == seed
                and row["official_score_by_seat"] == cash["score_by_seat"] == score
                and digest == official["transcript_sha256"] == row["raw_sha256"] == cash["logs_sha256"]
                and row["initial_terrain_u8_sha256"] == official["map_sha256"] == cash["official_map_sha256"] == terrain
                and official["steps"] == cash["terminal"] and official["statuses"] == ["DONE", "DONE"]
                and row["result"] == official["result"] == expected and abs(row["margin"] - margin) < 1e-7
                and result["header"]["pack_sha256"] == cash["pack_sha256"]):
            raise ValueError(f"conflicting score, terrain or tape identity in {name} seat {seat}")
        own, opponent = cash["per_game_totals"][seat], cash["per_game_totals"][1 - seat]
        times = row["search"]["time"]
        seats.append(dict(seat=seat, result=expected, scores=[score[seat], score[1-seat]],
                          margin=margin, turns=official["steps"], tape_sha256=digest,
                          searched=row["search_return_types"].get("searched", 0),
                          search_entries=row["search_invocations"], search_timer_median=median(times) if times else None,
                          own_cash_sources=own, opponent_cash_sources=opponent,
                          worst_ten_turns=cash["largest_10_turn_margin_decline"]))
    return dict(pack_sha256=result["header"]["pack_sha256"], seats=seats,
                wins=sum(r["result"] == "win" for r in seats), margin=mean(r["margin"] for r in seats))


def contrast(reference, candidate):
    changes = []
    for before, after in zip(reference["seats"], candidate["seats"]):
        player_changes = {}
        for side in ("own_cash_sources", "opponent_cash_sources"):
            player_changes[side] = {key: round(value - before[side][key], 3)
                                    for key, value in after[side].items()}
        changes.append(dict(seat=before["seat"], delta_margin=round(after["margin"] - before["margin"], 2),
                            **player_changes))
    return dict(seats=changes, delta_mean_margin=mean(r["delta_margin"] for r in changes))


if __name__ == "__main__":
    registration = read_json(HERE / "parent" / "registration.json")
    seed, terrain = registration["seed"], registration["initial_terrain_u8_sha256"]
    results = {key: summarize(key, seed, terrain) for key in ARMS}
    summary = dict(seed=seed, terrain_sha256=terrain, independent_maps=1, across_map_interval=None,
                   original_examiner_md5="57f9a594cce4f2ec9e3573d8d9d26ca1",
                   arms=results, trained_vs_matched_parent={
                       "single": contrast(results["parent"], results["candidate"]),
                       "hybrid": contrast(results["hybrid_parent"], results["hybrid_candidate"])})
    (HERE / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
