"""s2_paired -- per-game paired delta between two tags on identical (seed, side): A - B.
python3 s2_paired.py TAG_A TAG_B"""
import json, os, statistics as st, sys
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "evo_results.jsonl")
a, b = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(RES)]
A = {(r["seed"], r["policy_side"]): r for r in rows if r.get("tag") == a and "margin" in r}
B = {(r["seed"], r["policy_side"]): r for r in rows if r.get("tag") == b and "margin" in r}
k = sorted(set(A) & set(B))
d = [A[x]["margin"] - B[x]["margin"] for x in k]
print(json.dumps({"A": a, "B": b, "n": len(d), "delta_mean": round(st.mean(d), 1),
                  "delta_se": round(st.stdev(d) / len(d) ** 0.5, 1), "A_wins": sum(A[x]["win"] for x in k),
                  "B_wins": sum(B[x]["win"] for x in k), "A_worst": min(A[x]["margin"] for x in k),
                  "B_worst": min(B[x]["margin"] for x in k), "worst_delta": round(min(d)), "best_delta": round(max(d)),
                  "A_death": round(st.mean(A[x].get("pol_death_cargo") or 0 for x in k)),
                  "B_death": round(st.mean(B[x].get("pol_death_cargo") or 0 for x in k))}))
