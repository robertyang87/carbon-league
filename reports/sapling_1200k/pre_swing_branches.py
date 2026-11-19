"""Retrospective one-command alternatives three or five transitions before a cash swing."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import tree_audited_game as arena

MOVES = (None, "UP", "RIGHT", "DOWN", "LEFT")


def evaluate_branch(sim, root, trail, seat, override=None):
    sim.set_state(root)
    received = [0., 0.]
    losses = [0, 0]
    for offset, commands in enumerate(trail):
        joint = [dict(a) for a in commands]
        if offset == 0 and override:
            wid, move = override
            if move is None:
                joint[seat].pop(wid, None)
            else:
                joint[seat][wid] = move
        # Avoid silently sending a historical action to a now-missing unit.
        for player, cmd in enumerate(joint):
            if set(cmd) - set(sim.player_workers[player]) - {k for k, _ in sim.bases[player]}:
                return None
        sim.step(*joint)
        for _, victim, cargo, recipient, _ in sim.last_collisions:
            losses[victim] += 1
            if recipient == 1-victim:
                received[recipient] += cargo
        if offset + 1 < len(trail) and any(arena.exam._extinct(sim.get_state(), p) for p in (0,1)):
            return None
    return {"cash": list(sim.cash), "cargo": [sum(sim.workers[w].carbon for w in sim.player_workers[p]) for p in (0,1)],
            "trees": [len(sim.player_trees[p]) for p in (0,1)], "workers": [len(sim.player_workers[p]) for p in (0,1)],
            "received_from_enemy_collision": received, "lost_in_collisions": losses}


def audit_game(run, row):
    blob = (run / row["raw_file"]).read_bytes()
    if hashlib.sha256(blob).hexdigest() != row["raw_sha256"]:
        raise ValueError("transcript hash")
    raw = json.loads(gzip.decompress(blob))
    side = row["candidate_seat"]
    sim = arena.exam.FastCarbon(raw["game_configuration"]).reset(seed=row["seed"])
    states = [sim.get_state()]
    for actions in raw["actions"]:
        sim.step(*actions)
        states.append(sim.get_state())
    if arena.prior.hashed(states[-1]) != row["final_state_sha256"]:
        raise ValueError("trajectory digest")
    def relative(cash):
        return cash[1-side] - cash[side]
    # Earlier transition breaks exact ties. Use cash only, without the extinction scoring penalty.
    shock = max(range(1, len(states)), key=lambda t: relative(states[t]["cash"]) - relative(states[t-1]["cash"]))
    sweeps = []
    for horizon in (3,5):
        start = shock-horizon
        if start < 0:
            continue
        root = states[start]
        trail = raw["actions"][start:shock]
        base = evaluate_branch(sim, root, trail, side)
        if base is None or base["cash"] != states[shock]["cash"]:
            raise ValueError("open-loop control failed")
        branches = []
        rejected = 0
        for wid, worker in root["workers"].items():
            if worker["player"] != side or worker["type"] != "COLLECTOR":
                continue
            original = trail[0][side].get(wid)
            for move in MOVES:
                if move == original:
                    continue
                outcome = evaluate_branch(sim, root, trail, side, (wid, move))
                if outcome is None:
                    rejected += 1
                    continue
                branches.append({"unit": wid, "pos": worker["pos"], "root_cargo": worker["carbon"],
                                 "original": original, "alternative": move,
                                 "delta_own_cash": outcome["cash"][side]-base["cash"][side],
                                 "delta_enemy_cash": outcome["cash"][1-side]-base["cash"][1-side],
                                 "delta_own_margin": relative(base["cash"])-relative(outcome["cash"]),
                                 "delta_taken_from_enemy": outcome["received_from_enemy_collision"][side]-base["received_from_enemy_collision"][side],
                                 "delta_given_to_enemy": outcome["received_from_enemy_collision"][1-side]-base["received_from_enemy_collision"][1-side],
                                 "result":outcome})
        sweeps.append({"from_state":start, "horizon":horizon, "base":base, "altered_branches":branches,
                       "rejected_due_to_missing_future_actor_or_early_terminal":rejected})
    return {"run":run.name, "seed":row["seed"], "seat":side, "raw_sha256":row["raw_sha256"], "shock_end":shock,
            "cash_before":states[shock-1]["cash"], "cash_after":states[shock]["cash"],
            "opponent_one_turn_relative_cash_gain":relative(states[shock]["cash"])-relative(states[shock-1]["cash"]),
            "sweeps":sweeps}


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("runs",type=Path,nargs="+")
    cli.add_argument("--out",type=Path,required=True)
    args=cli.parse_args()
    rows=[]
    for run in args.runs:
        report=json.loads((run/"results.json").read_bytes())
        if report["status"]!="complete":
            raise ValueError("requires a completed run")
        rows.extend(audit_game(run,row) for row in report["games"])
    result={"scope":"Retrospective conditional branches on recorded opponent and later own commands. "
                    "Missing future actors or a newly premature terminal invalidate a branch. All alternatives are retained. "
                    "Extrema use hindsight and are not an estimate of an online policy's benefit.","games":rows}
    with args.out.open("xb") as handle:
        handle.write(gzip.compress(json.dumps(result,separators=(",",":"),allow_nan=False).encode(),mtime=0))
