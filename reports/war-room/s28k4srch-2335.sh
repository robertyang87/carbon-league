echo S28K4SRCH2335_BEGIN
date "+%F %T %Z"
echo "--- WRAPPER FILE ---"
md5sum /root/carbon/code/exp/k4v2r1.py /root/carbon/code/exp/k4v2g1.py
cat /root/carbon/code/exp/k4v2r1.py
echo "--- LAUNCHER LOG SEARCH MARKS ---"
wc -l /root/carbon/data/k4v2r1_e0.launcher.log
grep -n -i "search" /root/carbon/data/k4v2r1_e0.launcher.log | head -40 || echo NO-SEARCH-MARKS
echo "--- LAUNCHER LOG TAIL ---"
tail -20 /root/carbon/data/k4v2r1_e0.launcher.log
echo "--- PER-GAME RECORD KEYS + TIMING ---"
/root/carbon/.venv/bin/python - <<'PYEOF'
import json, statistics
p = "/root/carbon/data/bigeval_live/k4v2r1_e0.jsonl"
recs = [json.loads(l) for l in open(p) if l.strip()]
print("n_recs", len(recs))
print("keys", sorted(recs[0].keys()))
for k in ("tinc", "otinc", "wms", "worst_ms", "tms", "step_ms", "search", "nodes", "sims", "budget"):
    vals = [r[k] for r in recs if k in r and isinstance(r[k], (int, float))]
    if vals:
        print(f"{k}: n={len(vals)} min={min(vals)} mean={round(statistics.mean(vals),1)} max={max(vals)}")
    else:
        print(f"{k}: ABSENT")
# timing-ish fields present in every record?
print("tinc mean/max", round(statistics.mean(r.get("tinc", 0) for r in recs),1), max(r.get("tinc", 0) for r in recs))
print("margin mean", round(statistics.mean(r["margin"] for r in recs),1))
PYEOF
echo "--- SEARCH_GAMES DEFINITION IN CODE ---"
grep -n "search_games\|changed0" /root/carbon/code/bigeval.py | head -20
grep -n "SEARCH" /root/carbon/code/cnet2_search.py | head -20
echo "--- WEIGHTS ---"
md5sum /root/carbon/data/weights/cnet2_k4v2r1_fromstudio.npz /root/carbon/data/weights/cnet2_k4v2g1_fromstudio.npz
echo "--- CHAIN STATUS ---"
cat /root/carbon/data/k4probe.status
echo S28K4SRCH2335_END
