"""Apply the same tree-defense measurements to all five active Top 5 players.

Run with the active score-table snapshot, its replay directory, and the
expected ranks bound to this study. Rows without a replay are reported
explicitly and excluded from counts.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from elite_defense_evidence import new_totals, observe


RANKS = {
    "claude-AlphaCarbon-Master5s": 1,
    "claude-AlphaCarbon-Master6": 2,
    "RyanX-AIPowercore-it720-preview": 3,
    "RyanX-AIPowercore-M9XJ-630": 4,
    "ClaudeFable5.2-v7": 5,
}


def count_captures(steps, seat):
    timeline = [frame[0]["observation"]["players"] for frame in steps]
    return sum(len(before[seat][3].keys() & after[1-seat][3].keys())
               for before, after in zip(timeline, timeline[1:]))


def analyze(score_path, replays, out):
    score_bytes = score_path.read_bytes()
    matches = json.loads(score_bytes)
    totals = {name: new_totals() for name in RANKS}
    scheduled = dict.fromkeys(RANKS, 0)
    examples, checked, missing = {}, {}, []
    for game, match in sorted(matches.items()):
        names = [match["name_" + side] for side in match["order"]]
        if not set(names) & RANKS.keys():
            continue
        if game != f"{match['id']}-{match['game']}":
            raise ValueError(f"Game ID mismatch: {game}")
        for name in names:
            if name in scheduled:
                scheduled[name] += 1
        path = replays / (game + ".json.gz")
        if not path.exists():
            missing.append(game)
            continue
        raw = path.read_bytes()
        replay = json.loads(gzip.decompress(raw))
        steps = replay["steps"]
        if len(steps) < 2 or replay["rewards"] != [match["scores"][side] for side in match["order"]]:
            raise ValueError(f"Replay incomplete or inconsistent with official score: {game}")
        checked[game] = {"sha256": hashlib.sha256(raw).hexdigest(), "seats": {}}
        for seat, name in enumerate(names):
            if name not in RANKS:
                continue
            counts = observe(steps, seat, replay["configuration"], name, game,
                             names[1 - seat], totals[name], examples)
            expected = count_captures(steps, seat)
            if counts["capture"] != expected:
                raise ValueError(f"Recorded transfers disagree with transition audit: {game}")
            checked[game]["seats"][seat] = {"name": name, **counts}
    if not all(scheduled.values()):
        raise ValueError("A Top 5 player is absent from the supplied score table")
    result = {
        "cohort": "Active ranks 1-5 in the supplied November 18 2026 UTC score-table snapshot",
        "ranks": RANKS,
        "official_score_table_sha256": hashlib.sha256(score_bytes).hexdigest(),
        "scheduled_appearances": scheduled,
        "totals_by_model": totals,
        "complete": not missing,
        "analyzed_unique_games": len(checked),
        "missing_unique_games": missing,
        "examples": examples,
        "checked_games": checked,
        "units": "Distinct model appearances may share a game. One exposure is one non-expiring owned tree at one decision. Multiple rows in a game are correlated.",
        "near": "No ally on tree yet, at least one existing allied worker one cardinal move away. Ally entry is the actual chosen destination irrespective of survival.",
        "capture_check": "For each seat, total captured trees also counted independently as tree IDs changing from that seat to the opponent in adjacent observations.",
    }
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--score-rows", required=True, type=Path)
    cli.add_argument("--replays-dir", required=True, type=Path)
    cli.add_argument("--out", required=True, type=Path)
    a = cli.parse_args()
    analyze(a.score_rows, a.replays_dir, a.out)
