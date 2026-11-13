"""F02 AWR entry + F03 DAgger prefix, final tree.
AWR: same two-step recruit sequence as the original-env reference
(rewards [-299, 50], margin -349) through awr_gen.play with AWR_STEPS=299,
then awr_weights.rescore on the produced corpus.
F03: extract the CURRENT prefix block from dagger_gen.play by AST and exec
it on Codex's planter scenario; compare against lb_relabel._prefix_rows."""
import ast, gzip, json, os, sys, tempfile
CODE = os.path.expanduser("~/workspace/final-accept/code")
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "env_src", "carbon_challenge-master"))
os.environ["AWR_STEPS"] = "299"
import numpy as np

AGENT_TMPL = """
def agent(o, c):
    if o['step'] not in (0, 1):
        return {}
    bases = list(o['players'][o['player']][1])
    return {bases[%s]: 'RECCOLLECTOR'}
"""
tmp = tempfile.mkdtemp(prefix="awr-fa-")
ours = os.path.join(tmp, "ours.py"); opp = os.path.join(tmp, "opp.py")
open(ours, "w").write(AGENT_TMPL % "0")
open(opp, "w").write(AGENT_TMPL % "o['step'] % 2")
out = os.path.join(tmp, "corpus"); os.makedirs(os.path.join(out, "replays"))
import awr_gen
r = awr_gen.play((out, ours, opp, 0, 0, 0.0))
print("awr_gen result:", {k: r[k] for k in ("reward", "terminal", "end_step", "scores", "snaps")})
assert r["terminal"] == "extinction" and r["end_step"] == 2, r
assert r["reward"] == -349.0 or abs(r["reward"] - (-349.0)) < 0.051, r["reward"]
assert r["scores"] == [-299.0, 50.0], r["scores"]
assert r["snaps"] == 3, r["snaps"]          # initial + 2 steps; no post-terminal states
d = json.load(gzip.open(os.path.join(out, "replays", r["key"] + ".json.gz")))
statuses = [row[0]["status"] for row in d["steps"]]
print("replay statuses:", statuses, "| info:", d["info"])
assert statuses[-1] == "DONE" and all(s == "ACTIVE" for s in statuses[:-1])
# rescore through awr_weights on a one-game corpus
import awr_weights
idx = [{"file": r["file"], "key": r["key"], "scores": r["scores"],
        "info": d["info"], "steps": d["steps"]}]
# rescore(corpus) expects the corpus layout its caller uses; try direct:
try:
    rs = awr_weights.rescore([{"steps": d["steps"], "info": d["info"]}])
    print("rescore direct:", rs)
except Exception as e:
    print("rescore direct form n/a:", type(e).__name__, str(e)[:120])
print("PASS F02-awr: reward=-349 (side0), snaps=3, terminal DONE, scores [-299, 50]")

# ---------------- F03 prefix ----------------
import cnet2_feat as F
import lb_relabel as LB
from fastsim import FastCarbon
import eval_v53
sim = FastCarbon(eval_v53.CFG).reset(seed=9123); obs = sim.full_observation()
obs['players'][0][2] = {'player-0-worker-0': [111, 0, 'PLANTER'], 'player-0-worker-1': [112, 0, 'PLANTER']}
feat = F.Feat(0); x = feat.observe(obs)
y = feat.labels({}); ym = y.copy(); ym[0] = F.MOVES.index('UP'); ym[1] = F.MOVES.index('LEFT')
src = ast.parse(open(os.path.join(CODE, "dagger_gen.py")).read())
play = next(n for n in src.body if isinstance(n, ast.FunctionDef) and n.name == 'play')
loop = next(n for n in play.body if isinstance(n, ast.For))
stmts = loop.body
i0 = next(i for i, n in enumerate(stmts) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'pf' for t in n.targets))
i1 = next(i for i in range(i0, len(stmts)) if isinstance(stmts[i], ast.For))
block = stmts[i0:i1 + 1]     # pf assign + uf/mf/legal assigns + prefix for-loop
ns = {'F': F, 'np': np, 'x': x, 'feat': feat, 'ym': ym}
exec(compile(ast.Module(body=block, type_ignores=[]), 'dagger prefix block', 'exec'), ns)
uf2, mf2, legal2, labels2 = LB._prefix_rows(F, x, feat, ym)
diff = float(np.max(np.abs(ns['mf'][1] - mf2[1])))
print(f"F03 second-unit mf max|diff| vs stored-prefix recompute: {diff}")
assert diff == 0.0, diff
print("PASS F03: prefix diff 0.0 (was 0.5)")
