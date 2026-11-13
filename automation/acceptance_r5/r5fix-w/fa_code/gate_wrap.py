"""gate_wrap -- same pass rule as gate.py, for a prebuilt wrapper file.
  python3 gate_wrap.py WRAPPER.py TAG [N_SEEDS=32] [SEED0=900000]"""
import json, os, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from evo import run_pairs
from gate import V9
def main(wrap, tag, n_seeds=32, seed0=900000):
    rows = run_pairs([(wrap, V9, seed0, n_seeds)], tag=f"gate_{tag}", log=False)
    m = [r["margin"] for r in rows if "margin" in r]; err = sum(1 for r in rows if "error" in r)
    n = len(m); mean = st.mean(m); se = st.stdev(m) / n ** .5; wr = sum(x > 0 for x in m) / n
    wms = max(r.get("a_worst_step_ms", 0) for r in rows if "margin" in r)
    big = n >= 64 and wr >= .65 and mean >= 1000 and mean - 2 * se > 0 and min(m) > -5000
    alt = n >= 30 and wr > .80
    rec = {"tag": tag, "wrapper": wrap, "seed0": seed0, "n": n, "err": err, "wins": sum(x > 0 for x in m),
           "winrate": round(wr, 3), "mean": round(mean, 1), "se": round(se, 1), "worst": round(min(m), 1),
           "worst_ms": wms, "pass": err == 0 and wms < 500 and (big or alt), "margins": [round(x) for x in m]}
    open(os.path.join(HERE, "..", "data", "gate_results.jsonl"), "a").write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "margins"}), flush=True)
if __name__ == "__main__":
    a = sys.argv[1:]; main(a[0], a[1], int(a[2]) if len(a) > 2 else 32, int(a[3]) if len(a) > 3 else 900000)
