"""record_shadow_ds_traces -- record the shadow DeepSeek eval games
into traces/ in the standard carbon-trace/1 format and verify final
cash against eval_shadow_v2_results.jsonl (the 核对差 discipline).

Agent loading matches eval_shadow_v2.py exactly (shadow via
make_agent(), policy_v0_5 via make_agent(), example / reference_juzi
via module agent), so a recorded game is bit-identical to the eval
game. Frame capture / file format / naming reuse record_trace.py.

Usage: python3 record_shadow_v2_traces.py EVAL_JSONL SHADOW_FILE TAG
"""

import gzip
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))

from fastsim import FastCarbon  # noqa: E402
from eval_v0_5 import CFG  # noqa: E402
import record_trace as RT  # noqa: E402

_counter = [100]


def load_agent_fn(path, kind):
    _counter[0] += 1
    name = f"recds_{kind}_{_counter[0]}"
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    if kind == "policy":
        return m.make_agent()
    return m.agent


def record_game(shadow_file, opp_kind, opp_file, seed, shadow_side, tag):
    sim = FastCarbon(CFG).reset(seed=seed)
    shadow = load_agent_fn(os.path.join(HERE, shadow_file), "policy")
    opp = load_agent_fn(os.path.join(HERE, opp_file),
                        "policy" if opp_kind == "policy_v0_5" else "opp")
    agents = [None, None]
    agents[shadow_side] = shadow
    agents[1 - shadow_side] = opp
    frame0, st0 = RT.snapshot(sim)
    bases = {str(p): [[bid, pos] for bid, pos in st0["bases"][p]]
             for p in range(2)}
    frames = [frame0]
    for _ in range(300):
        acts = []
        for p in range(2):
            o = sim.obs(p)
            acts.append(agents[p](o, CFG))
        sim.step(acts[0], acts[1])
        frame, _ = RT.snapshot(sim)
        frame["acts"] = [
            {k: v for k, v in (a or {}).items() if v is not None}
            for a in acts]
        frames.append(frame)
    pol_name = os.path.splitext(os.path.basename(shadow_file))[0]
    trace = {
        "meta": {
            "format": "carbon-trace/1",
            "policy": pol_name,
            "opp": opp_kind,
            "tag": tag,
            "seed": seed,
            "policy_side": shadow_side,
            "steps": 300,
            "size": int(CFG["size"]),
            "bases": bases,
            "final_cash": frames[-1]["cash"],
            "legend": {
                "workers": "wid -> [pos_index, cargo, C=COLLECTOR|P=PLANTER]",
                "trees": "tid -> [pos_index, owner_player, age]",
                "pos_index": "row * size + col, row 0 = top (obs index space)",
                "cash": "cash IS the score in this game",
            },
        },
        "frames": frames,
    }
    return trace


def main():
    eval_jsonl, shadow_file, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    expected = {}
    with open(eval_jsonl) as f:
        for line in f:
            r = json.loads(line)
            if r["shadow_file"] != shadow_file:
                continue
            expected[(r["opp"], r["seed"], r["shadow_side"])] = (
                r["shadow_cash"], r["opp_cash"])
    rows = []
    for (opp_kind, seed, side), (exp_my, exp_opp) in sorted(
            expected.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        opp_file = {"policy_v0_5": "policy_v0_5.py",
                    "example": "example.py",
                    "reference_juzi": "reference_juzi.py",
                    "base_v53": "base_v53.py"}[opp_kind]
        # same global-random seeding as eval_shadow_ds.py (opponents
        # draw from the global random module)
        import random as _random
        _random.seed(7919 * seed + 104729 * (1 - side) + 31)
        trace = record_game(shadow_file, opp_kind, opp_file, seed, side,
                            tag)
        fname = RT.trace_filename(shadow_file, opp_kind, seed, side)
        path = os.path.join(RT.TRACES_DIR, fname)
        RT.write_trace(trace, path)
        fc = trace["meta"]["final_cash"]
        my, op = fc[side], fc[1 - side]
        d_my, d_op = round(my - exp_my, 2), round(op - exp_opp, 2)
        rows.append({
            "policy": trace["meta"]["policy"], "opp": opp_kind,
            "seed": seed, "side": side, "my": my, "opp_cash": op,
            "exp_my": exp_my, "exp_opp": exp_opp,
            "diff_my": d_my, "diff_opp": d_op, "file": fname,
        })
        print(f"{fname}: {my:.2f} : {op:.2f} "
              f"(eval {exp_my:.2f} : {exp_opp:.2f}) "
              f"diff {d_my:+.2f}/{d_op:+.2f}", flush=True)
    with open(os.path.join(HERE, "record_shadow_ds_index_rows.json"),
              "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("rows -> record_shadow_ds_index_rows.json")


if __name__ == "__main__":
    main()
