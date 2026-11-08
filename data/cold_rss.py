import sys, os, time, resource, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'env_src', 'carbon_challenge-master'))
import eval_v53
from fastsim import FastCarbon
path = sys.argv[1]
src = open(path).read()
random.seed(1); sim = FastCarbon(eval_v53.CFG).reset(seed=970700)
t0 = time.time(); g = {"__name__": "submission"}; exec(compile(src, "submission.py", "exec"), g); t_load = time.time() - t0
fa = g["agent"]; fb = eval_v53.load_agent_fn(os.path.join(os.path.dirname(path), "submission_cnet2_v13.py"))
ts = []
for step in range(1, 301):
    t = time.time(); a0 = fa(sim.obs(0), eval_v53.CFG); ts.append(time.time() - t)
    a1 = fb(sim.obs(1), eval_v53.CFG); sim.step(a0, a1)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)   # macOS: bytes
print(f"exec {t_load*1000:.0f} ms | first call {ts[0]*1000:.0f} ms | mean {sum(ts)/len(ts)*1000:.0f} ms | worst {max(ts)*1000:.0f} ms | last {ts[-1]*1000:.0f} ms | peak RSS {rss:.0f} MiB (process incl. v13 opponent)")
