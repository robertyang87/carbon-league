"""Replay the live guarded candidate on its completed games and count changed V5 actions."""
import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "code"))
from build_sapling_cash_dagger import module
from fastsim import FastCarbon
from v5_guard_replay import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    index = json.loads(args.results.read_bytes())
    if index["status"] != "complete" or len(index["games"]) != 4:
        raise ValueError("wait for all four scored games")
    models = ROOT / "data/weights/sapling_1200k"
    pack = models / "soft_it50_v5_guard_agent.py"
    student = module(models / "soft_it50_v5_agent.py", "exam_plain_v5")
    guarded = module(pack, "exam_guarded_v5")
    rows = []
    for g in index["games"]:
        raw_bytes = (args.results.parent / g["raw_file"]).read_bytes()
        assert sha(raw_bytes) == g["raw_sha256"]
        r = json.loads(gzip.decompress(raw_bytes))
        assert r["pack_sha256"] == sha(pack.read_bytes())
        seat = r["candidate_seat"]
        sim = FastCarbon(r["game_configuration"]).reset(seed=r["seed"])
        changed_steps = []
        for t, joint in enumerate(r["actions"]):
            obs = sim.obs(seat)
            deployed = guarded.agent(obs,sim.config)
            assert deployed == joint[seat], (r["seed"], seat, t)
            if student.agent(obs,sim.config) != deployed:
                changed_steps.append(t)
            sim.step(*joint)
        stats = dict(guarded.agent.last_stats)
        assert stats["changed_roots"] == len(changed_steps)
        rows.append(dict(seed=r["seed"], seat=seat, verified_calls=len(r["actions"]),
                         raw_sha256=g["raw_sha256"], stats=stats, changed_steps=changed_steps))
    result = dict(pack_sha256=sha(pack.read_bytes()), source_results_sha256=sha(args.results.read_bytes()),
                  scope="Same observed guarded trajectories for both selectors. No estimate of long-horizon counterfactual uplift.",
                  games=rows)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
