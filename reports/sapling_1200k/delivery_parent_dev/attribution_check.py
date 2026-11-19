"""Replay the archived games with the deterministic hybrid and verify its action counters."""
import gzip
import json
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
sys.path.insert(0, str(RUN.parent))
import tree_audited_game as arena


def check(row, pack):
    prior = arena.prior
    tape = RUN / row["raw_file"]
    if prior.digest(tape) != row["raw_sha256"]:
        raise ValueError(f"archive digest mismatch: {tape}")
    raw = json.loads(gzip.decompress(tape.read_bytes()))
    if prior.digest(pack) != raw["pack_sha256"]:
        raise ValueError("candidate identity mismatch")
    seed, seat = row["seed"], row["candidate_seat"]
    sim = arena.exam.FastCarbon(raw["game_configuration"]).reset(seed=seed)
    candidate = prior.shared.module(pack, "replay_candidate")
    for turn, commands in enumerate(raw["actions"]):
        replayed = candidate.agent(sim.obs(seat), sim.config)
        if replayed != commands[seat]:
            raise ValueError(f"action mismatch at {seed}/{seat}/{turn}")
        sim.step(*commands)
    if sim.step_count != row["actual_transitions"] or prior.hashed(sim.get_state()) != row["final_state_sha256"]:
        raise ValueError(f"terminal state mismatch: {seed}/{seat}")
    return {"map": seed, "seat": seat, "matched_action_steps": sim.step_count,
            "raw_sha256": row["raw_sha256"], "counts": candidate._COUNTS}


if __name__ == "__main__":
    report = json.loads((RUN / "results.json").read_bytes())
    pack = ROOT / "data/weights/sapling_1200k/hybrid_it50_v3.py"
    identities = {(g["seed"], g["candidate_seat"]) for g in report["games"]}
    expected = {(seed, seat) for seed in (620224310, 620224311) for seat in (0, 1)}
    if not report["games"] or len(report["games"]) != len(expected) or identities != expected:
        raise ValueError("incomplete map/seat record")
    rows = [check(game, pack) for game in report["games"]]
    print(json.dumps({"pack_sha256": arena.prior.digest(pack), "games": rows}, indent=2))
