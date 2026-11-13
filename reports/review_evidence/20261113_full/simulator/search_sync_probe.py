"""Production search control flow; fixed network outputs, real FastCarbon/Feat.

Constructed positions isolate terminal and budget boundaries. This does not
measure natural-game frequency or neural policy quality. CARBON_REPO chooses
the checkout; every run creates a fresh output directory.
"""
import ast
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types


def find_repository():
    explicit = os.environ.get("CARBON_REPO")
    if explicit:
        root = pathlib.Path(explicit).expanduser().resolve()
        if not (root / "code/fastsim.py").is_file():
            raise RuntimeError("CARBON_REPO must contain code/fastsim.py")
        return root
    for start in (pathlib.Path(__file__).resolve().parent, pathlib.Path.cwd()):
        for root in (start, *start.parents):
            if (root / "code/fastsim.py").is_file():
                return root
    raise RuntimeError("Set CARBON_REPO to the repository checkout")


ROOT = find_repository()
sys.path.insert(0, str(ROOT / "code"))
import numpy as np
import cnet2_search as MAIN
import cnet2_feat as F
from fastsim import FastCarbon


def source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PACK = source_module(ROOT / "code/packs/cnet2_search.py", "review_packed_search")


def helpers(path):
    tree = ast.parse(path.read_text())
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "make_search_agent")
    return [n for n in factory.body if isinstance(n, ast.FunctionDef)
            and n.name in ("_extinct", "_terminal", "leaf", "_search", "_mcts")]


def classify(sim):
    return [not any(w.player == p for w in sim.workers.values())
            and not any(t.player == p for t in sim.trees.values())
            and (not sim.bases[p] or sim.cash[p] < 30) for p in (0, 1)]


def run(source, name, *, step=100, delayed=False, min_depth=3,
        roll_move="NONE", clock_values=(), horizon=5, mode="sweep"):
    mod = MAIN if source == "main" else PACK
    path = ROOT / ("code/cnet2_search.py" if source == "main" else "code/packs/cnet2_search.py")
    sim = FastCarbon().reset(full_carbon=[0] * 225)
    obs = sim.full_observation()
    obs["step"] = step
    for p in (0, 1):
        obs["players"][p][0] = 20
    base = sim.bases[1][0][1]
    apos = base + (30 if delayed else 15)
    obs["players"][0][2] = {"player-0-worker-0": [apos, 0, "COLLECTOR"]}
    obs["players"][1][2] = {"player-1-worker-0": [apos - 1, 100, "COLLECTOR"]}
    f, fo = F.Feat(0), F.Feat(1)
    x, xo = f.observe(obs), fo.observe(dict(obs, player=1))
    y = np.zeros(F.OWN, np.int64)
    y[0] = F.MOVES.index("UP")
    yo = np.zeros(F.OWN, np.int64)
    yo[0] = F.MOVES.index("LEFT")  # Seat-1 canonical LEFT is world RIGHT.

    class FixedPolicy:
        last_enc = (np.zeros((1, 1)), np.zeros(1))

        def enc_for(self, *a):
            return self.last_enc

        def act(self, *a, **kw):
            return yo.copy(), None, None

    def decode(pol, X, aux, tok, ctx, forced=None, **kw):
        if forced is None:
            if pol is None:  # MCTS rollout: our rows have temperature zero.
                lab = np.zeros((len(aux), F.OWN), np.int64)
                lab[np.asarray(kw["temp"]) == 0, 0] = F.MOVES.index(roll_move)
                return lab
            return np.repeat(yo[None], len(aux), axis=0)
        return np.stack([np.where(a >= 0, a, y) for a in forced])

    def rollout(pol, xs, auxs, teachers):
        lab = np.zeros((len(xs), F.OWN), np.int64)
        lab[:len(xs) // 2, 0] = F.MOVES.index(roll_move)
        return lab

    ticks = iter(clock_values)
    cfg = dict(mod.DEFAULTS, MAX_BR=2, MIN_DEPTH=min_depth, H=horizon,
               LOG=True, BUDGET=100, HARD=100)
    if mode == "mcts":
        cfg.update(ROOT_S=1, STYLES=(1,), TEMPS=(1.0,), CLOCK="sims", FIX_SIMS=(2,))
    stats = dict.fromkeys(("searched", "branches", "branch_steps", "depth_sum",
                          "shallow", "changed", "gain"), 0)
    log = []
    ns = dict(cfg=cfg, st={"f": f, "fo": fo, "T_now": 1, "ot": 1, "oenc": None},
              np=np, F=F, pol=FixedPolicy(), rpol=None, pool=[], stats=stats,
              log=log, T=1, key_units=mod.key_units, decode_b=decode,
              act_b=rollout, FastCarbon=FastCarbon, feat_copy=mod.feat_copy,
              clk=lambda: next(ticks, 0), time=types.SimpleNamespace(time=lambda: 0))
    if mode == "mcts":
        ns["st"].update(lg0=np.zeros((F.OWN, 5)), ow=np.ones(1), ti=0,
                        oenc=(mod.stack_x([xo]), np.zeros((1, 1, 1)), np.zeros((1, 1))))
        ns.update(xpol=None, XT=1, stack_x=mod.stack_x,
                  ctx_b=lambda pol, pre, teachers: np.zeros((len(teachers), 1)),
                  encode_b=lambda pol, X, teachers: (np.zeros((len(teachers), 1, 1)), np.zeros((len(teachers), 1))))
    exec(compile(ast.Module(body=helpers(path), type_ignores=[]), str(path), "exec"), ns)
    got = ns["_mcts" if mode == "mcts" else "_search"](obs, x, y, {"player-0-worker-0": "UP"}, xo, 299-step, 0, 0)
    rows = []
    for s in ns["pool"]:
        extinct = classify(s)
        depth = stats["depth_sum"] if mode == "sweep" else s.step_count-step
        leaf_args = (s, 0, max(0, 299-step-depth))
        value = ns["leaf"](*leaf_args, [20, 20])
        rows.append(dict(step=s.step_count, cash=list(s.cash), extinct=extinct,
                         terminal=s.step_count >= 299 or any(extinct), leaf=float(value)))
    return dict(name=name, source=source, mode=mode, returned_actions=got, stats=stats,
                branches=rows, log=log)


cases = [
    ("all_terminal_root_default_depth", {}),
    ("all_terminal_root_depth1_diagnostic", dict(min_depth=1)),
    ("all_terminal_at_depth2", dict(delayed=True, roll_move="UP")),
    ("mixed_budget_before_rollout", dict(delayed=True, clock_values=(100,))),
    ("mixed_budget_after_one_rollout", dict(delayed=True, clock_values=(0, 1, 100))),
    ("mixed_reaches_min_depth", dict(delayed=True, horizon=3)),
    ("natural_horizon_mixed_extinction", dict(step=298, delayed=True)),
]
out = {"commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
       "main": [], "pack": [], "mcts": [], "source_hashes": {}}
for name, kw in cases:
    r = run("main", name, **kw)
    expected_shallow = int(name.startswith("mixed_budget"))
    assert r["stats"]["shallow"] == expected_shallow, r
    assert r["returned_actions"] == ({"player-0-worker-0": "UP"} if expected_shallow else {}), r
    if name == "all_terminal_at_depth2":
        assert r["stats"]["depth_sum"] == 2 and [b["step"] for b in r["branches"]] == [102, 101], r
    if name == "mixed_budget_before_rollout":
        assert r["stats"]["depth_sum"] == 1 and [b["terminal"] for b in r["branches"]] == [False, True], r
    if name == "mixed_budget_after_one_rollout":
        assert r["stats"]["depth_sum"] == 2 and [b["step"] for b in r["branches"]] == [102, 101], r
    if name == "mixed_reaches_min_depth":
        assert [b["step"] for b in r["branches"]] == [103, 101], r
    if name == "natural_horizon_mixed_extinction":
        assert [b["step"] for b in r["branches"]] == [299, 299], r
        assert [b["leaf"] for b in r["branches"]] == [0, 22], r
    r["pass"] = True
    out["main"].append(r)

# Re-run old failing pack cases with corrected expectations against actual current source.
for name, kw in [
    ("all_terminal_root_cut_by_budget", dict(clock_values=(100,))),
    ("terminal_branches_freeze", {}),
    ("natural_horizon_stops_at_299", dict(step=298, delayed=True)),
]:
    out["pack"].append(run("pack", name, **kw))
assert out["pack"][2]["returned_actions"] == {}
assert out["pack"][0]["stats"]["shallow"] == 0
assert out["pack"][0]["returned_actions"] == {}
assert [b["step"] for b in out["pack"][1]["branches"]] == [101, 101]
assert [b["step"] for b in out["pack"][2]["branches"]] == [299, 299]
for source in ("main", "pack"):
    for name, kw, expected_steps in [
        ("mcts_root_terminal", {}, [101, 101]),
        ("mcts_later_terminal", dict(delayed=True, roll_move="UP"), [102, 101]),
        ("mcts_mixed_horizon", dict(step=298, delayed=True), [299, 299]),
    ]:
        r = run(source, name, mode="mcts", **kw)
        assert r["returned_actions"] == {} and r["stats"]["shallow"] == 0, r
        assert [b["step"] for b in r["branches"]] == expected_steps, r
        r["pass"] = True
        out["mcts"].append(r)
for path in ("code/cnet2_search.py", "code/packs/cnet2_search.py", "code/fastsim.py", "code/packs/fastsim.py", "code/cnet2_feat.py", "code/packs/cnet2_feat.py"):
    out["source_hashes"][path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
base = os.environ.get("CARBON_REVIEW_OUT")
if base:
    pathlib.Path(base).expanduser().mkdir(parents=True, exist_ok=True)
    base = str(pathlib.Path(base).expanduser().resolve())
dest = pathlib.Path(tempfile.mkdtemp(prefix="carbon-search-r5-", dir=base)) / "search-boundaries.json"
dest.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
print(f"Fresh evidence: {dest}", file=sys.stderr)
