"""Evaluate two frozen team orders using only their acting seat's observed history."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INTAKE = HERE.parent
ACTOR = INTAKE.parents[1]
CRITIC = ACTOR / "joint_response_critic_dev"
sys.path.insert(0, str(INTAKE))
import select_orders as selection


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


def critic_for_root():
    path = CRITIC / "candidate.py"
    check(sha(path) == "14146394cb35db682fa98bbdae1a4abd2f54cd1256988fbcee70ae9ff1dda08b", "critic drift")
    spec = importlib.util.spec_from_file_location("archived_joint_critic", path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model


def measure(ticket, views, cfg):
    model = critic_for_root()
    t = ticket["root_step"]
    votes = {}
    for index, observation in enumerate(views):
        check(index == observation["step"] and observation["player"] == ticket["seat"], "history mismatch")
        if index == 0:
            model.agent(observation, cfg)
        elif index < t:
            model._FEAT.observe(observation)
        else:
            check(index == t and selection.common.digest(observation) == ticket["root_public_sha256"], "root mismatch")
            break
    else:
        raise ValueError("missing root")

    original = model._jrc_attach

    def capture(x, aux, enc, baseline):
        ctx = model._jrc_context(observation, cfg, model._FEAT, x, aux, enc[1][1])
        menu = model._jrc_options(ctx, observation, cfg, model._FEAT, x, aux, baseline)
        orders = [model._FEAT.to_actions(labels) for labels, _, _ in menu]
        scores = model._jrc_scores(np.stack([entry[1] for entry in menu]), model._JRC_WEIGHTS)
        check(np.isfinite(scores).all() and len(orders) == len(set(selection.common.digest(a) for a in orders)), "invalid menu")
        votes["original_baseline"] = orders[0]
        votes["menu_size"] = len(orders)
        votes["orders"] = {}
        for name, order in ticket["orders"].items():
            labels = model._FEAT.labels(order)
            check(order == model._FEAT.to_actions(labels), "command cannot round trip")
            representation = model._jrc_repr(ctx, observation, cfg, model._FEAT, x, aux, labels)
            index = orders.index(order) if order in orders else None
            votes["orders"][name] = {"order": order, "menu_index": index, "prefix_representable": representation is not None,
                                     "physical_menu_score": None if index is None else float(scores[index])}
        return original(x, aux, enc, baseline)

    model._jrc_attach = capture
    chosen = model.agent(observation, cfg)
    check(chosen == model._JRC_LAST["chosen"] and votes["original_baseline"] == model._JRC_LAST["parent"], "public critic differs")
    votes["original_final_choice"] = chosen
    pair = votes["orders"]
    valid = all(r["physical_menu_score"] is not None for r in pair.values())
    delta = pair["alternative"]["physical_menu_score"] - pair["parent"]["physical_menu_score"] if valid else None
    votes["pair_score_difference"] = delta
    votes["higher_arm"] = None if not valid or delta == 0 else "alternative" if delta > 0 else "parent"
    return votes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out
    roots = INTAKE / "roots.json"
    check(sha(roots) == "94034e2de67e9bc92cc765f8acdc923cdab678749167a51623a270051bdab78b", "registered roots drift")
    rows = []
    for ticket in json.loads(roots.read_bytes())["tickets"]:
        src, identity = selection.source_for(ticket["seed"])
        check(all(ticket[k] == v for k, v in identity.items()), "signed source drift")
        cfg = {k: v for k, v in src["configuration"].items() if k != "randomSeed"}
        views = (row["public"] for row in src["rows"] if row["step"] <= ticket["root_step"])
        rows.append({**identity, "t": ticket["root_step"], "public_sha256": ticket["root_public_sha256"],
                     **measure(ticket, views, cfg)})
    result = {"critic": sha(CRITIC / "candidate.py"), "weights": sha(CRITIC / "weights.npz"), "roots": sha(roots),
              "script": sha(Path(__file__)), "rows": rows}
    raw = (json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    if out.exists():
        check(out.read_bytes() == raw, "preexisting result differs")
    else:
        out.write_bytes(raw)
    print([(r["seed"], [v["menu_index"] for v in r["orders"].values()], r["pair_score_difference"]) for r in rows])
    print(sha(out))


if __name__ == "__main__":
    main()
