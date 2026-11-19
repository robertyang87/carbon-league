"""Audit the terminal-trained blend on candidate trajectories after selection."""
import gzip
import json
from pathlib import Path
import sys

import train


def main():
    run = train.HERE / "results.json"
    result = json.loads(run.read_bytes())
    if result.get("status") != "complete":
        raise ValueError("training has not completed")
    chosen = result["chosen"]
    deployed = train.load_agent(train.PARENT, "chosen_terminal_blend", train.packed(train.ARMS[chosen]))
    original = train.load_agent(train.PARENT, "original_hybrid")
    no_student = train.load_agent(train.PARENT, "zero_sapling_weight", train.packed([[0.,0.,0.],[0.,0.,0.]]))
    rows = []
    for g in result["games"]:
        if g["stage"] != "holdout" or g["arm"] != chosen: continue
        path = train.HERE / g["raw_file"]
        if train.prior.digest(path) != g["raw_sha256"]:
            raise ValueError("changed holdout tape")
        raw = json.loads(gzip.decompress(path.read_bytes()))
        sim = train.exam.FastCarbon(raw["game_configuration"]).reset(seed=g["seed"])
        seat = g["candidate_seat"]
        studied, changed, student = [], [], []
        for t, joint in enumerate(raw["actions"]):
            obs = sim.obs(seat)
            if t % 12 == 0 or t+1 == len(raw["actions"]):
                outputs = [train.act(net,obs,sim.config) for net in (deployed,original,no_student)]
                if outputs[0] != joint[seat]: raise ValueError("frozen actor disagrees with tape")
                studied.append(t)
                if outputs[0] != outputs[1]: changed.append(t)
                if outputs[0] != outputs[2]: student.append(t)
            else:
                for mod, deps in (deployed,original,no_student):
                    sys.modules.update(deps)
                    mod._FEAT.observe(obs)
            sim.step(*joint)
        rows.append(dict(seed=g["seed"], seat=seat, audited_steps=studied,
                         differs_from_parent=changed, differs_from_zero_sapling=student))
    train.prior.shared.atomic_json(train.HERE / "attribution.json", {
        "chosen": chosen, "recipe_sha256": train.prior.digest(run), "games": rows,
        "scope": "Every twelfth candidate holdout root and final action, using all previous public observations. "
                 "Disabling the student changes its coefficient to zero and restores unit expert weight."})


if __name__ == "__main__":
    main()
