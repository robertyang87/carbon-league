"""Check that hidden replay extras and a late comparison cannot alter a public Sapling order."""

import copy
from pathlib import Path
import importlib.util
import json
import numpy as np
import bridge
import build
from qualify import package, norm


class LateClock:
    def __init__(self, mode):
        self.mode = mode
        self.calls = 0

    def perf_counter(self):
        self.calls += 1
        return (
            (0.0, 0.55, 0.56, 0.57, 0.59)[min(4, self.calls - 1)]
            if self.mode == "skip"
            else (0.0, 0.1, 0.2, 0.73, 0.83)[min(4, self.calls - 1)]
        )


def verify():
    current = bridge.load(bridge.HERE / "release.json")
    path = build.DEFAULT
    bridge.require(bridge.sha(path) == current["sha256"], "candidate changed")
    data = bridge.load(bridge.HERE / "qualification.json.gz")
    samples = [
        (game, x) for game in data["cases"] for x in game["opportunities"] if x["status"] == "completed"
    ]
    game, choice = samples[0]
    raw = bridge.load(bridge.ROOT / game["source"])
    cfg = raw["game_configuration"]
    seat = game["seat"]
    sim = bridge.FastCarbon(cfg).reset(seed=raw["seed"])
    clean = package(path, "plain")
    dirty = package(path, "with_extra_replay_truth")
    results = {}
    for t in range(choice["t"] + 1):
        obs = sim.obs(seat)
        extra = dict(
            obs,
            full_carbon=[87.0] * 225,
            oracle_carbon=[92.0] * 225,
            reveal={cell: 96.0 for cell in range(225)},
            future_opponent_action={"not_public": "UP"},
        )
        a = clean.agent(obs, cfg)
        b = dirty.agent(extra, cfg)
        bridge.require(
            norm(a) == norm(b)
            and np.array_equal(clean._FEAT.est, dirty._FEAT.est)
            and np.array_equal(clean._BR_OP.est, dirty._BR_OP.est),
            "hidden root or history leakage",
        )
        if t == choice["t"]:
            bridge.require(
                clean._BR_LAST["status"] == dirty._BR_LAST["status"] == "completed",
                "selected diagnostic root missing",
            )
            bridge.require(
                clean._BR_LAST["comparison"]["samples"] == dirty._BR_LAST["comparison"]["samples"],
                "different rollout path through hidden data",
            )
            results["public_invariance"] = {
                "turns_through": t,
                "focal_action": a,
                "full_joint_and_both_beliefs_equal_steps": t + 1,
                "root_samples_equal": True,
            }
        else:
            sim.step(*raw["actions"][t])
    for mode, wanted in (("skip", "budget_skip"), ("late", "budget_fallback")):
        tester = package(path, "clock_" + mode)
        sim = bridge.FastCarbon(cfg).reset(seed=raw["seed"])
        for t in range(choice["t"] + 1):
            if t == choice["t"]:
                tester.time = LateClock(mode)
            got = tester.agent(sim.obs(seat), cfg)
            if t == choice["t"]:
                probe = tester._BR_LAST
                bridge.require(
                    probe["status"] == wanted and norm(got) == norm(probe["base"]),
                    "budget fallback changed actual command",
                )
                results[mode] = {
                    "status": wanted,
                    "same_unsearched_order": True,
                    "entry_time_observations": tester.time.calls,
                    "possesses_complete_round": "comparison" in probe,
                }
            else:
                sim.step(*raw["actions"][t])
    results.update(
        art_sha256=current["sha256"],
        probe_source_sha256=bridge.sha(Path(__file__)),
        qualification_sha256=bridge.sha(bridge.HERE / "qualification.json.gz"),
        raw_sha256=bridge.sha(bridge.ROOT / game["source"]),
    )
    bridge.save(bridge.HERE / "isolation.json", results)
    print(results, flush=True)


if __name__ == "__main__":
    verify()
