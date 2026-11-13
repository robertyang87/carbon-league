"""pack_timing PACK [N]: single process, N games vs Master2 real; every call of the pack timed (incl. the first-step load)."""
import os, sys, time, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_v53 as E
PACK = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
T = []; first = []
_orig = E.load_agent_fn
def load(path):
    f = _orig(path)
    if not path.endswith(PACK): return f
    def g(o, c):
        t = time.perf_counter(); a = f(o, c); d = time.perf_counter() - t; T.append(d)
        if o["step"] == 0: first.append(d)
        return a
    return g
E.load_agent_fn = load
for i in range(N):
    E.play_game(("t", PACK, "exp/opp_master2r.py", 995000 + i, i % 2))
T.sort(); q = lambda p: T[min(len(T) - 1, int(p * len(T)))]
print(f"{PACK}: {len(T)} steps; median {1000*statistics.median(T):.1f} ms, p99 {1000*q(0.99):.1f} ms, max {1000*T[-1]:.1f} ms, steps>0.95s {sum(t > 0.95 for t in T)}, >1s {sum(t > 1.0 for t in T)}; first-step median {1000*statistics.median(first):.0f} ms")
