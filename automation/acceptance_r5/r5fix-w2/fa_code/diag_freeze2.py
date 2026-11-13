"""Freeze diagnostic v2: a FREEZE step = collector stationary, off own
base, cargo >= 100, and cargo UNCHANGED across the step (not collecting,
not depositing -- pure local-minimum paralysis of a loaded returner).

Usage: python3 diag_freeze2.py POLICY_FILE OPP SEED SIDE
Prints final stranded cargo, total freeze steps, longest freeze run and
all freeze runs of length >= 4 (wid, start step, length, cargo).
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))
from fastsim import FastCarbon  # noqa: E402

_spec = json.load(open(os.path.join(
    HERE, "env_src", "carbon_challenge-master",
    "zerosum_env", "envs", "carbon", "carbon.json")))
CFG = {}
for _k, _v in _spec["configuration"].items():
    CFG[_k] = _v.get("default") if isinstance(_v, dict) else _v


def load(path, kind):
    s = importlib.util.spec_from_file_location(
        f"fz2_{kind}_{os.getpid()}_{abs(hash(path)) % 99999}", path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m.make_agent() if kind == "policy" else m.agent


def main():
    policy_file, opp_kind = sys.argv[1], sys.argv[2]
    seed, side = int(sys.argv[3]), int(sys.argv[4])
    sim = FastCarbon(CFG).reset(seed=seed)
    pol = load(os.path.join(HERE, policy_file), "policy")
    opp = load(os.path.join(HERE, f"{opp_kind}.py"), "opp")
    agents = [None, None]
    agents[side] = pol
    agents[1 - side] = opp
    bases = {pos for _, pos in sim.bases[side]}
    runs = {}   # wid -> [cur_len, start_step, cargo]
    events = []
    total_freeze = 0
    for t in range(300):
        pre = {w: (d["pos"], d["carbon"]) for w, d in sim.get_state()["workers"].items()
               if d["player"] == side and d["type"] == "COLLECTOR"}
        sim.step(agents[0](sim.obs(0), CFG), agents[1](sim.obs(1), CFG))
        post = sim.get_state()["workers"]
        for wid, (pos, cargo) in pre.items():
            cur = runs.setdefault(wid, [0, -1, 0.0])
            frozen = (wid in post and post[wid]["pos"] == pos
                      and abs(post[wid]["carbon"] - cargo) < 0.05
                      and cargo >= 100 and pos not in bases)
            if frozen:
                if cur[0] == 0:
                    cur[1], cur[2] = t, cargo
                cur[0] += 1
                total_freeze += 1
            else:
                if cur[0] >= 4:
                    events.append((wid, cur[1], cur[0], round(cur[2], 1)))
                cur[0] = 0
    for wid, cur in runs.items():
        if cur[0] >= 4:
            events.append((wid, cur[1], cur[0], round(cur[2], 1)))
    st = sim.get_state()
    stranded = sum(w["carbon"] for w in st["workers"].values() if w["player"] == side)
    print(json.dumps({
        "policy": policy_file, "opp": opp_kind, "seed": seed, "side": side,
        "cash": [round(c, 1) for c in st["cash"]],
        "stranded": round(stranded, 1),
        "freeze_steps": total_freeze,
        "longest_freeze_run": max((e[2] for e in events), default=0),
        "freeze_events>=4": sorted(events, key=lambda e: -e[2]),
    }))


if __name__ == "__main__":
    main()
