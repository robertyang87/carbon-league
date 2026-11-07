"""Load a submission the way a string-exec platform would (no __file__, fresh globals) and play one game."""
import sys, time
sys.path.insert(0, '/Users/baggioustb/dev/carbon/carbon-league/code')
import eval_v53
from fastsim import FastCarbon
src = open(sys.argv[1]).read()
g = {"__name__": "__main__", "__builtins__": __builtins__}
t = time.time(); exec(compile(src, "<submission>", "exec"), g); print("load ok", round(time.time() - t, 2), "s")
sim = FastCarbon(eval_v53.CFG).reset(seed=int(sys.argv[2]) if len(sys.argv) > 2 else 970500)
opp = eval_v53.load_agent_fn('exp/submission_nn_v9.py'); w = 0
for s in range(300):
    t = time.time(); a = g["agent"](sim.obs(0), eval_v53.CFG); w = max(w, time.time() - t)
    sim.step(a, opp(sim.obs(1), eval_v53.CFG))
st = sim.get_state(); print("cash", [round(c) for c in st["cash"]], "worst ms", round(w * 1000))
