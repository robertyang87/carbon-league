#!/usr/bin/env python3
"""ppou2 telemetry sidecar: endgame residual-tree counts + endgame unsold
cargo (strategy-group rulings 2026-11-15 13:12 trees / 13:59 cargo).
Runs ON L20 with cwd=/root/carbon/probe_m2/code.

For the first N seeds (ascending seed order, no cherry-picking) of an exam
field, re-run each game with the PINNED eval_v53.play_game (unmodified) and,
after the game, read the terminal full observation from the game sim:
counts[p] = len(observation.players[p][3])  -- whole-dict count, no
owner-prefix filtering (scout caliber, reproduced 2.941 / 5.0 / 4 games on
the 26-game ladder corpus).
cargo[p] = sum(entry[1] for entry in observation.players[p][2].values())
-- each worker's carried carbon (entry index 1); players[p][1] is the
static recruitment-center dict (bid -> pos), no inventory. Cargo caliber
reproduced value-by-value on the 51-game Echo4 ladder corpus
(echo4_pull51, oldest-first numbering): #46 361.79, #49 143.54,
#42 115.64, 哈基米 v2 pro 279.14, 哈基米 v3 399.64 -- all five match the
scout's registered numbers exactly (Echo4 identified by name, seat via
the match record's order field).

Identity gate (hard): every re-run's margin/win must equal the field's
per-game line exactly; any mismatch -> gate failure, caller must void the
sidecar and report. The game sim is captured by subclassing FastCarbon in
eval_v53's namespace only; no pinned file is modified (harness 354bcd1b and
eval_v53 45792dfb untouched).

Usage: remote_sidecar_trees.py <a_file> <b_file> <field_jsonl> <n> <out_jsonl>
a_file/b_file are relative to the exam dir, as in bigeval task tuples.
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())  # exam dir: eval_v53 / fastsim live there
import eval_v53

REG = []
_orig_fc = eval_v53.FastCarbon


class _Rec(_orig_fc):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        REG.append(self)


def main():
    a_file, b_file, field_path, n, out_path = (
        sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    lines = {}
    with open(field_path) as f:
        for ln in f:
            d = json.loads(ln)
            if d.get("err") is None:
                lines[d["seed"]] = d
    seeds = sorted(lines)[:n]
    eval_v53.FastCarbon = _Rec
    out = open(out_path, "w")
    gate_fail = 0
    for seed in seeds:
        ln = lines[seed]
        side = ln["side"]
        del REG[:]
        res = eval_v53.play_game(("sidecar", a_file, b_file, seed, side))
        sim = REG[0]
        obs = sim.full_observation()
        counts = [len(obs["players"][0][3]), len(obs["players"][1][3])]
        cargo = [round(sum(w[1] for w in obs["players"][p][2].values()), 2)
                 for p in (0, 1)]
        ok = (round(res["margin"], 2) == ln["margin"] and res["win"] == ln["win"]
              and sim.step_count == res["nsteps"])
        if not ok:
            gate_fail += 1
        rec = {"seed": seed, "side": side, "margin": res["margin"],
               "win": res["win"], "field_margin": ln["margin"],
               "field_win": ln["win"], "gate_ok": ok,
               "cand_trees": counts[side], "opp_trees": counts[1 - side],
               "trees_p0": counts[0], "trees_p1": counts[1],
               "cand_cargo": cargo[side], "opp_cargo": cargo[1 - side],
               "cargo_p0": cargo[0], "cargo_p1": cargo[1],
               "end_step": res["nsteps"]}
        out.write(json.dumps(rec) + "\n")
        out.flush()
    out.close()
    print(json.dumps({"sampled": len(seeds), "gate_fail": gate_fail}))
    return 1 if gate_fail else 0


if __name__ == "__main__":
    sys.exit(main())
