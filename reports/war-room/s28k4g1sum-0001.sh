echo S28K4G1SUM0001_BEGIN
date "+%F %T %Z"
/root/carbon/.venv/bin/python - <<'PYEOF'
import json
recs = [json.loads(l) for l in open("/root/carbon/data/bigeval.jsonl") if l.strip()]
for r in recs:
    if r["tag"] == "k4v2g1_e0":
        print("SCALARS", json.dumps({k: r[k] for k in ("tag","a","b","seed0","n","err","wins","mean","se","lo2se","worst","worst_ms","dep","tree","deathcargo","plants","t","search_games","changed0_games")}, ensure_ascii=False))
        g = r["games"]
        print("GAMES_N", len(g))
        for i in range(0, len(g), 20):
            print("GAMES", i, json.dumps(g[i:i+20], separators=(",", ":")))
PYEOF
echo S28K4G1SUM0001_END
