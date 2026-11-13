"""cnet2_search_worst -- per-game breakdown of an eval: worst game, #< -500, #< -1000, and for the worst K seeds
a replay with ledgers of the candidate AND of v12 in the same seat/seed (did the search make it worse?).
python3 cnet2_search_worst.py TAG [K]"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import eval_v53
tag = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
rec = [json.loads(l) for l in open(os.path.join(HERE, "..", "data", "bigeval.jsonl")) if '"%s"' % tag in l][-1]
g = rec["games"]; m = [x[2] for x in g]
print(json.dumps({"tag": tag, "n": len(m), "worst": min(m), "lt500": sum(x < -500 for x in m), "lt600": sum(x < -600 for x in m), "lt1000": sum(x < -1000 for x in m)}))
KEYS = ("deposits", "tree_income", "plant_count", "seize_count", "recruit_count", "deaths", "death_cargo")
def brief(r):
    lp, lo = r["ledger_pol"], r["ledger_opp"]
    return {"margin": r["margin"], "pol": {k: lp[k] for k in KEYS}, "opp": {k: lo[k] for k in KEYS}, "comp": r["comp"]}
for seed, side, mg in sorted(g, key=lambda x: x[2])[:K]:
    r1 = eval_v53.play_game((tag, rec["a"], rec["b"], seed, side))
    r0 = eval_v53.play_game(("v12ref", "exp/submission_cnet2_v12.py", rec["b"], seed, side))
    print(json.dumps({"seed": seed, "side": side, "eval_margin": mg, "replay": brief(r1), "v12_same_seed": brief(r0)}), flush=True)
