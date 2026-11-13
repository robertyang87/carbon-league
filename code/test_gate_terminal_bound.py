"""Regression fixtures for the teacher gate's first-terminal bound.

The du sampler used to bound root steps by the replay FILE length, but
replay files are padded with copies of the terminal frame: roots sampled
into the padding are terminal scenes whose continuations short-circuit
(teacher_calls=0, du=0) and silently dilute du_mean / sigma / N. These
fixtures pin the fixed semantics:
- first_terminal_step agrees with k0_aux_labels.first_terminal (the K0
  data gates' reference implementation) on every synthetic case;
- the sampler never emits a root at/past a game's first-terminal frame
  and stamps each spec with that frame;
- _drop_padding_rows removes exactly the stamped padding rows from du
  aggregation and keeps unstamped (legacy) rows.

Run: python test_gate_terminal_bound.py  (from code/, or an acceptance tree)
"""
import gzip
import json
import os
import sys
import tempfile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LEARNER_THREADS"):
    os.environ[name] = "1"
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_HERE, "stage1_teacher_gate.py")):
    CODE = _HERE
else:
    CODE = os.path.abspath(os.path.join(_HERE, "../../code"))
sys.path.insert(0, CODE)

import stage1_teacher_gate as G
import k0_aux_labels as K0

RES = []


def check(name, ok):
    RES.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name, flush=True)


def frame(status0="ACTIVE", status1="ACTIVE", players=None):
    e0 = {"status": status0}
    if players is not None:
        e0["observation"] = {"players": players}
    return [e0, {"status": status1}]


HEALTHY = [100.0, ["b1"], ["u1"], ["t1"]]
STARVED = [10.0, ["b1"], [], []]  # no units/trees, cash below cheapest recruit


def ft_cases():
    cases = {}
    cases["end_of_record"] = [frame() for _ in range(12)]
    steps = [frame() for _ in range(12)]
    steps[5] = frame(status1="DONE")
    cases["status_mid"] = steps
    steps = [frame() for _ in range(12)]
    steps[0] = frame(status0="DEAD")
    cases["status_first_frame"] = steps
    steps = [frame(players=[HEALTHY, HEALTHY]) for _ in range(12)]
    steps[3] = frame(players=[STARVED, HEALTHY])
    cases["extinction"] = steps
    steps = [frame(players=[HEALTHY, HEALTHY]) for _ in range(12)]
    steps[4] = frame(status0="DONE", players=[STARVED, HEALTHY])
    cases["status_beats_extinction_same_frame"] = steps
    steps = [frame(players=[HEALTHY, HEALTHY]) for _ in range(12)]
    steps[2] = frame(players=[[100.0, [], [], []], HEALTHY])  # no bases: extinct
    cases["extinction_no_bases"] = steps
    return cases


def write_replay(path, steps):
    with gzip.open(path, "wt") as fh:
        json.dump({"steps": steps}, fh)


def main():
    for name, steps in ft_cases().items():
        got = G.first_terminal_step(steps)
        want = K0.first_terminal(steps)
        check(f"first_terminal agrees with K0 reference: {name}", got == want)
    check("empty frames carry no terminal evidence (gate-side extension)",
          G.first_terminal_step([[] for _ in range(10)]) == (9, "end_of_record"))

    with tempfile.TemporaryDirectory() as td:
        steps_a = [frame() for _ in range(300)]
        steps_a[210] = frame(status0="DONE")
        pa = os.path.join(td, "gameA.json.gz")
        write_replay(pa, steps_a)
        pb = os.path.join(td, "gameB.json.gz")
        write_replay(pb, [frame() for _ in range(300)])

        specs = []
        for seed in range(20):
            specs.extend(G.sample_scene_specs([pa, pb], [(200, 240)], seed))
        a = [s for s in specs if s["game"] == "gameA"]
        b = [s for s in specs if s["game"] == "gameB"]
        check("sampler emits roots for the partially-live layer", len(a) > 0 and len(b) > 0)
        check("no root at/past first terminal (gameA ft=210)",
              all(s["step"] < 210 for s in a) and all(s["ft"] == 210 for s in a))
        check("all-live game sampled across the whole layer (gameB)",
              all(200 <= s["step"] < 240 for s in b) and all(s["ft"] == 299 for s in b))

        wide = []
        for seed in range(10):
            wide.extend(G.sample_scene_specs([pa], [(0, 300)], seed))
        check("wide layer still bounded by first terminal",
              len(wide) > 0 and all(s["step"] < 210 for s in wide))

    rows = [{"game": "g", "step": 209, "ft": 210, "du": 0.1},
            {"game": "g", "step": 210, "ft": 210, "du": 0.0},
            {"game": "g", "step": 250, "ft": 210, "du": 0.0},
            {"game": "g", "step": 100, "du": 0.2},
            {"game": "g", "step": 100, "ft": None, "du": 0.3}]
    used, dropped = G._drop_padding_rows(rows)
    check("aggregation drops exactly the stamped padding rows",
          [r["step"] for r in used] == [209, 100, 100]
          and [r["step"] for r in dropped] == [210, 250])

    print(f"==== {sum(RES)}/{len(RES)} checks passed ====", flush=True)
    return 0 if all(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
