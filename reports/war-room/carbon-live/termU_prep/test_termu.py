import math, re, types
import numpy as np

src = open('/home/hatch/workspace/carbon-live/termU_prep/rl2_ppo_train.py.u').read()
start = src.index('\nDRAW_W = 0.5\n')
end = src.index('def eval_tasks')
ns = {'np': np, 'math': math}
exec(src[start:end], ns)
game_rewards, terminal_u = ns['game_rewards'], ns['terminal_u']

def ref_u(m):
    w = 1.0 if m > 0 else (0.5 if m == 0 else 0.0)
    return 0.5 * w + 0.5 / (1.0 + math.exp(-m / 1000.0))

def mk_g(margin, T=300):
    dm = np.full(T, margin / T)
    return {"dmargin": dm, "dworth": np.zeros(T), "dep": np.zeros(T),
            "lost": np.zeros(T), "margin": float(margin)}

args = types.SimpleNamespace(reward="u", rscale=1.0, shape_iters=200,
                             c_dep=0.0, c_lost=0.0, win_bonus=0.1)
fails = []
for m in (1000.0, 250.0, 1.0, 0.0, -1.0, -250.0, -1000.0, 1655.0):
    r = game_rewards(mk_g(m), args, 80)
    expect = 2.0 * (ref_u(m) - 0.5)
    if r.dtype != np.float32: fails.append(f"dtype {m}")
    if not np.all(r[:-1] == 0): fails.append(f"nonterminal nonzero {m}")
    if abs(float(r[-1]) - expect) > 1e-6: fails.append(f"value {m}: {r[-1]} vs {expect}")
    if abs(float(r.sum()) - expect) > 1e-6: fails.append(f"sum {m}")
# antisymmetry for decisive results
for m in (1000.0, 250.0, 1655.0):
    rp = game_rewards(mk_g(m), args, 80)[-1]; rm = game_rewards(mk_g(-m), args, 80)[-1]
    if abs(float(rp) + float(rm)) > 1e-6: fails.append(f"antisym {m}")
# win_bonus must not leak into u mode (args carries 0.1 on purpose)
r = game_rewards(mk_g(0.0), args, 80)
if float(r[-1]) != 0.0: fails.append("draw not exactly 0 / win_bonus leaked")
# old modes untouched: worth path still dworth-based
args2 = types.SimpleNamespace(reward="worth", rscale=3.33e-4, shape_iters=200,
                              c_dep=0.0, c_lost=0.0, win_bonus=0.1)
g = mk_g(300.0); g["dworth"] = np.ones(300)
r = game_rewards(g, args2, 80)
if abs(float(r[:-1].sum()) - 299 * 3.33e-4) > 1e-5: fails.append("worth path changed")
print("U(+1000)=%.6f r=%+.6f | U(0)=%.4f r=%+.4f | U(-1000)=%.6f r=%+.6f" % (
    terminal_u(1000.0), game_rewards(mk_g(1000.0), args, 80)[-1],
    terminal_u(0.0), game_rewards(mk_g(0.0), args, 80)[-1],
    terminal_u(-1000.0), game_rewards(mk_g(-1000.0), args, 80)[-1]))
print("TESTS", "ALL PASS" if not fails else "FAIL: " + "; ".join(fails))
