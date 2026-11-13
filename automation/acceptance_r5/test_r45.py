"""Residual 4: in-process play() must reject a game whose search fallback
swallowed root_errors (agent = bare function, stats in __globals__, no
.stats attribute -- the acceptance probe shape). Residual 5 is checked
via --help in the shell, not here."""
import ast, importlib.util, os, sys
CODE = "/home/hatch/workspace/fix4-residual/tree"
sys.path.insert(0, CODE)
import numpy as np

spec = importlib.util.spec_from_file_location(
    "spg", os.path.join(CODE, "producer", "selfplay_gen.py"))
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)

fails = []
def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f" | {extra}" if extra else ""))
    if not cond: fails.append(name)

def recruit(obs, cfg, different=False):
    if obs["step"] not in (0, 1): return {}
    return {list(obs["players"][obs["player"]][1])[obs["step"] % 2 if different else 0]: "RECCOLLECTOR"}
def opp(obs, cfg): return recruit(obs, cfg, True)
def noop(obs, cfg): return {}

# bare wrapper agent exec'd from the real make_search_agent AST (no factory
# tail -> no .stats attribute; counters live in its globals)
tree = ast.parse(open(os.path.join(CODE, "cnet2_search.py")).read())
factory = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "make_search_agent")
wrapper = [x for x in factory.body if isinstance(x, ast.FunctionDef) and x.name == "agent"][-1]
def inner_error(*a, **kw): raise ValueError("injected policy inference failure")
ns = {"_inner": inner_error, "stats": {}, "cfg": {}}
exec(compile(ast.Module(body=[wrapper], type_ignores=[]), "wrapper", "exec"), ns)
bare_agent = ns["agent"]
check("probe shape: bare agent has no .stats attribute", getattr(bare_agent, "stats", None) is None)

outdir = "/home/hatch/workspace/fix4-residual/r4_out"
import shutil
if os.path.exists(outdir): shutil.rmtree(outdir)
gen = G._Gen("stub.npz", dict(G.CONF), "t-a", "t-b")
gen.agents = [bare_agent, recruit]
entry, n, margin, err = gen.play(0, outdir)
check("in-process play() rejects fallback-polluted game", entry is None and err is not None and "policy_errors" in err,
      f"err={err}")
check("root_errors were actually swallowed by fallback", ns["stats"].get("root_errors") == 2,
      str(ns["stats"].get("root_errors")))
fpath = os.path.join(outdir, "replays", "selfplay-0.json.gz")
check("no DONE replay written for rejected game", not os.path.exists(fpath))

# production shape: agent WITH .stats attribute must be counted exactly once
class AttrAgent:
    def __init__(self): self.stats = {}
    def __call__(self, obs, cfg):
        self.stats["root_errors"] = self.stats.get("root_errors", 0) + 1
        return {}
gen2 = G._Gen("stub.npz", dict(G.CONF), "t-a", "t-b")
gen2.agents = [AttrAgent(), noop]
entry2, n2, m2, err2 = gen2.play(1, outdir)
check("attribute-stats agent counted (not missed, not doubled into a pass)",
      entry2 is None and err2 is not None and "root_errors" in err2, f"err={err2}")

# normal game unaffected
gen3 = G._Gen("stub.npz", dict(G.CONF), "t-a", "t-b")
gen3.agents = [noop, noop]
entry3, n3, m3, err3 = gen3.play(2, outdir)
check("normal game still produced", entry3 is not None and err3 is None and n3 == 299,
      f"n={n3} err={err3}")
check("normal game replay written", os.path.exists(os.path.join(outdir, "replays", "selfplay-2.json.gz")))

print("RESULT:", "ALL PASS" if not fails else f"FAILURES {fails}")
sys.exit(1 if fails else 0)
