"""Measure where the frozen orchard policy abstains on exact public development roots."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import tree_audited_game as arena

HERE = Path(__file__).resolve().parent
PACK = HERE.parents[1] / "data/weights/sapling_1200k/orchard_it50_v3.py"
SHA = "1dc7345ea71ab692e3db09c00b0958f5fc086302493c5301afa89eb97125b38d"


def game(run, row):
    m = arena.prior.shared.module(PACK, "orchard_funnel")
    tape = run / row["raw_file"]
    if arena.prior.digest(tape) != row["raw_sha256"]:
        raise ValueError("wrong action tape")
    raw = json.loads(gzip.decompress(tape.read_bytes()))
    seat = row["candidate_seat"]
    sim = arena.exam.FastCarbon(raw["game_configuration"]).reset(seed=row["seed"])
    original = m._orchard_decide
    groups = {}
    changed = 0

    def observe(F, x, prefix, slot, normal, logits, legal, changes):
        if slot >= 10 or x["tok_kind"][slot] != 1 or changes:
            return original(F, x, prefix, slot, normal, logits, legal, changes)
        pos = int(x["tok_pos"][slot])
        cell = int(F.CANON[pos]) if seat else pos
        item = m.orchard_features(m._ORCHARD_OBS, m._ORCHARD_OBS["carbon"],
                                  seat, cell, m._ORCHARD_OBS["step"], m._ORCHARD_CFG)
        if item is None:
            return original(F, x, prefix, slot, normal, logits, legal, changes)
        kind, price, features = item
        key = kind + ("_would_buy" if normal == 0 else "_would_move")
        count = groups.setdefault(key, Counter())
        count["with_opportunity"] += 1
        if prefix.cash < price:
            count["insufficient_cash"] += 1
            return original(F, x, prefix, slot, normal, logits, legal, changes)
        count["eligible"] += 1
        net = m.orchard_predict(m._ORCHARD_MODELS[kind], features) - price
        safe = legal & (prefix.reserved[F.DEST[pos]] == 0) & (prefix.undecided[F.DEST[pos]] == 0)
        safe &= ~np.isfinite(prefix.a["en_min"][F.DEST[pos]]) & (prefix.a["en_pl"][F.DEST[pos]] == 0)
        if normal == 0:
            safe[0] = False
            if net < -25:
                count["below_minus25"] += 1
                if safe.any():
                    count["also_safe_alternative"] += 1
                    alt = int(np.argmax(np.where(safe, logits, -np.inf)))
                    if logits[0] - logits[alt] < min(3., -net/65.):
                        count["also_within_logit_gate"] += 1
        else:
            # Report both marginal counts and nested intersections in deployment order.
            count["above_plus50"] += net > 50
            count["safe_stay"] += bool(safe[0])
            count["reserve_90"] += prefix.cash >= price+90
            if net > 50 and safe[0]:
                count["high_and_safe"] += 1
                if prefix.cash >= price+90:
                    count["high_safe_reserve"] += 1
                    if logits[normal] - logits[0] < min(2., net/100.):
                        count["also_within_logit_gate"] += 1
        proposed = original(F, x, prefix, slot, normal, logits, legal, changes)
        count["pre_joint_check_changes"] += proposed != normal
        return proposed

    m._orchard_decide = observe
    for turn, commands in enumerate(raw["actions"]):
        played = m.agent(sim.obs(seat), sim.config)
        changed += played != commands[seat]
        if run.name == "orchard_dev" and played != commands[seat]:
            raise ValueError(f"frozen decision differs on candidate tape at {turn}")
        sim.step(*commands)
    if arena.prior.hashed(sim.get_state()) != row["final_state_sha256"]:
        raise ValueError("trajectory failed terminal identity")
    return {"seed":row["seed"], "seat":seat, "turns":sim.step_count, "groups":groups,
            "different_joint_actions_from_tape":changed, "frozen_counts":m._ORCHARD_COUNTS}


def main():
    if arena.prior.digest(PACK) != SHA:
        raise ValueError("unexpected candidate")
    result = {"pack_sha256":SHA, "definition":"One eligible count per proposed decoder planter slot with a public purchase opportunity and enough prefix cash. Overlapping marginal counts are named independently; ratio interpretation uses each named parent count.", "runs":{}}
    for run in map(Path,sys.argv[1:]):
        rows = json.loads((run/"results.json").read_bytes())["games"]
        result["runs"][run.name] = [game(run, row) for row in rows]
    print(json.dumps(result, indent=2, default=int))


if __name__ == "__main__":
    main()
