"""Profile matched held decisions and independently capture the first enemy order produced by the frozen parent search."""
import gzip
import json
import os
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

import model as small
import study


def conditional_panels(rows, weights):
    seeds = sorted({row["seed"] for row in rows})
    mode_ids = {"native": "native", "prior": "prior", "new": str(weights["beta"])}
    panels = {}
    for seed in seeds:
        rs = [row for row in rows if row["seed"] == seed]
        stats = {name: study.stats(rs, mode, weights) for name, mode in mode_ids.items()}
        panels[str(seed)] = {
            name: {key: summary["totals"][key] for key in ("correct", "workers", "heavy_correct", "heavy", "exact", "roots")}
            for name, summary in stats.items()
        }
    coverage = {}
    for size in (1,2,3,4):
        rs = [r for r in rows if len(r["menu"]) == size]
        coverage[str(size)] = {
            "roots": len(rs),
            "any_exact": sum(any(a["exact"] for a in r["metrics"]) for r in rs),
            **{name: study.stats(rs, mode, weights)["totals"]["exact"] for name, mode in mode_ids.items()},
        }
    return panels, coverage


def verified_parent_order(m, obs, est, prev, cfg, own, sw, sb):
    """Stop at the native transition boundary, after native generation and opponent sampling."""
    class FirstPair(Exception):
        pass

    captured = []
    original_turn = m._turn

    def capture(root, configuration, pair):
        captured.append(pair[1-obs["player"]])
        raise FirstPair()

    try:
        m._turn = capture
        m._search(obs, cfg, est, own, sw, sb, prev, time.perf_counter())
    except FirstPair:
        pass
    finally:
        m._turn = original_turn
    return captured[0] if captured else None


def by_phase(records, key):
    out = {}
    for phase in records[0][key]:
        seconds = [r[key][phase] for r in records]
        out[phase] = dict(zip(("p50", "p95", "p99", "max"), np.percentile(seconds, [50,95,99,100]).tolist()))
    return out


def main():
    here = Path(__file__).resolve().parent
    rows = study.read(here / "held_roots.json.gz")
    weight = study.read(here / "weights.json")
    indexed = {(r["seed"],r["seat"],r["t"]): r for r in rows}
    maps, coverage = conditional_panels(rows, weight)
    cards = [c for c in study.read(study.D/"opponent_rollout_calibration/protocol.json")["tickets"] if c["split"] == "held"]
    m = study.old.initialize()
    config = study.read(study.D.parents[1]/"code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json")["configuration"]
    cfg = {k:v["default"] for k,v in config.items() if isinstance(v,dict) and "default" in v}
    cfg["episodeSteps"] = 300
    profile = []
    flip_logged = Counter()
    replayed = []
    fields = ("menu", "opponent_values", "native_logit_sums", "common_self_model_order")
    for card in cards:
        tape,_ = study.old.read(card)
        seat = card["seat"]
        with zipfile.ZipFile(study.D/card["rel"]) as z:
            turns = [json.loads(s) for s in z.read(card["arm"]+"/turns.jsonl").splitlines()]
        tracker = m.BeliefTracker(m._Bel, seat)
        prev = None
        for t,(obs,pair) in enumerate(tape):
            estimate,_ = tracker.step(obs["players"],t,obs["carbon"])
            if (card["seed"],seat,t) in indexed:
                resp = turns[t]["calls"][1-seat]["response"]
                flip_logged["enemy_search_counts_present"] += "search_counts" in resp
                flip_logged["held_roots"] += 1
            if t in (8,50,98,146,194,242,290):
                raw = indexed[(card["seed"],seat,t)]
                entry = dict(seed=card["seed"],seat=seat,step=t)
                # Do not change inference, only its elapsed time clock. Alternate first clock per root.
                clocks = [("wall", time.perf_counter),("cpu",time.process_time)]
                if len(profile)%2:
                    clocks.reverse()
                for name, clock in clocks:
                    small.perf_counter = clock
                    try:
                        alt = small.infer(m,obs,estimate,prev,cfg)
                    finally:
                        small.perf_counter = time.perf_counter
                    assert all(raw[k] == alt[k] for k in fields)
                    entry[name] = alt["timings"]
                # Verify the captured parent order on old held-map roots across the two opposing seats.
                if t in (50,194):
                    x = m._quant(m.encode(obs["players"], t, seat, estimate, [1.]*m.N, prev)).astype(np.float32)
                    sw, sb, _ = m._net()(x[None],m._SID)
                    captured = verified_parent_order(m,obs,estimate,prev,cfg,alt["common_self_model_order"],sw[0],sb[0])
                    if captured is not None:
                        assert captured == raw["menu"][0]
                    replayed.append({"seed":card["seed"],"seat":seat,"step":t,"native_search_reached_enemy_draw":captured is not None,
                                     "actual_parent_opponent_order":captured,"model_index0":raw["menu"][0]})
                profile.append(entry)
            tracker.set_actions(obs["players"], pair[seat])
            prev = obs["players"]
    summary = dict(
        held_by_seed=maps, exact_coverage_by_menu=coverage, original_opponent_telemetry=dict(flip_logged),
        timing=dict(matched_roots=len(profile),affinity=sorted(os.sched_getaffinity(0)),torch_threads=m._torch.get_num_threads(),
                    wall=by_phase(profile,"wall"),process_cpu=by_phase(profile,"cpu"),verified_replays=len(profile)*2),
        native_search_captures=replayed, raw_profile_sha=None,
    )
    study.save(here/"timing_roots.json.gz",profile)
    summary["raw_profile_sha"] = study.sha((here/"timing_roots.json.gz").read_bytes())
    study.save(here/"verification.json",summary)
    print(json.dumps(summary["timing"],indent=2))

if __name__=="__main__":
    main()
