"""Reproduce the two-step Master6 return-to-base case from a frozen Git tree.

Run with Python + NumPy from any directory:
  python reports/master_family/deep_review/replay_return_counterfactual.py
Optional --out writes the JSON receipt. Both replay and simulator are loaded
from REV, so later working-tree code changes cannot silently alter the test.
This holds all other recorded actions fixed; it is not a closed-loop match.
"""
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

REV = "37965ea578851b5f3ff88ffc42d7119ff716b5cc"
MATCH = "d2c7d061739bef723dc4abfffb22de2a"
REPLAY = f"data/master6_corpus/{MATCH}-1.json.gz"
WORKER = "player-1-worker-61"
HERE = Path(__file__).resolve().parent


def blob(path):
    return subprocess.check_output(["git", "-C", str(HERE), "show", f"{REV}:{path}"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    raw = blob(REPLAY)
    replay = json.loads(gzip.decompress(raw))
    matches = json.loads(blob("data/master6_matches.json"))
    match = next(m for m in matches if m["id"] == MATCH and m["game"] == 1)
    names = [match["name_" + side] for side in match["order"]]
    assert names[1] == "claude-AlphaCarbon-Master6"
    assert replay["rewards"] == [match["scores"][s] for s in match["order"]]

    simulator = blob("code/fastsim.py")
    results = []
    with tempfile.TemporaryDirectory(prefix="master6-return-") as temporary:
        source = Path(temporary) / "frozen_fastsim.py"
        source.write_bytes(simulator)
        spec = importlib.util.spec_from_file_location("frozen_fastsim", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for changed in (False, True):
            sim = module.FastCarbon(replay["configuration"])
            sim.load_observation(replay["steps"][297][0]["observation"])
            trajectory = []
            for state in (298, 299):
                actions = [copy.deepcopy(r["action"] or {}) for r in replay["steps"][state]]
                if changed:
                    actions[1][WORKER] = "RIGHT"
                sim.step(*actions)
                actual = sim.full_observation()
                expected = replay["steps"][state][0]["observation"]
                assert actual["step"] == state
                if not changed:
                    assert actual["players"] == expected["players"], state
                trajectory.append({
                    "state": state,
                    "cash_by_seat": list(sim.cash),
                    "margin_master6": sim.cash[1] - sim.cash[0],
                    "baseline_players_exact": actual["players"] == expected["players"] if not changed else None,
                    "worker61": copy.deepcopy(actual["players"][1][2].get(WORKER)),
                })
            results.append({"changed_worker61_to_right": changed, "trajectory": trajectory})

    original = results[0]["trajectory"][-1]
    modified = results[1]["trajectory"][-1]
    assert round(original["margin_master6"], 2) == -38.37
    assert round(modified["margin_master6"], 2) == -6.33
    receipt = {
        "revision": REV, "replay": REPLAY, "names_by_seat": names,
        "replay_sha256": hashlib.sha256(raw).hexdigest(),
        "simulator_sha256": hashlib.sha256(simulator).hexdigest(),
        "scope": "Two steps from state 297; all other recorded actions fixed; no opponent replanning.",
        "results": results,
        "master6_cash_gain": round(modified["cash_by_seat"][1] - original["cash_by_seat"][1], 2),
        "outcome_flipped": modified["margin_master6"] > 0,
    }
    output = json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
