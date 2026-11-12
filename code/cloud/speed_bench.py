"""speed_bench: single process, 20 self-play games each (median s/game; per-side agent ms/step), then cProfile of one Grove1 game."""
import cProfile, io, os, pstats, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AWR_US_CONF", '{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}')
import eval_v53 as E
def run(a, b, n=20):
    w = []; ms = []
    for i in range(n):
        t = time.time(); r = E.play_game(("b", a, b, 995000 + i, i % 2)); w.append(time.time() - t)
        ms.append((r["a_ms_per_step"] + r["b_ms_per_step"]) / 2)
    return statistics.median(w), statistics.median(ms), r["nsteps"]
for a in ("exp/opp_master2r.py", "awr_us_env.py"):
    s, m, ns = run(a, a); print(f"{a}: median {s:.2f} s/game (both sides); agent {m:.1f} ms/step per side over {ns} steps; simulator+ledger {s - 2*m*ns/1000:.2f} s/game", flush=True)
pr = cProfile.Profile(); pr.enable(); E.play_game(("p", "awr_us_env.py", "awr_us_env.py", 995100, 0)); pr.disable()
for key in ("cumulative", "tottime"):
    o = io.StringIO(); pstats.Stats(pr, stream=o).sort_stats(key).print_stats(20); print(f"== cProfile by {key}"); print(o.getvalue()[-6000:])
