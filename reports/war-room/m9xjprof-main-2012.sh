#!/bin/bash
# M9XJ it260 teacher profile MAIN (stage 2): per-game plants/residual from replay tree-id transitions.
# Definitions mirror eval_v53 ledger: plants = tree ids appearing in teacher's set that were in NEITHER set
# at previous recorded state; seized = ids that moved from opponent's set; residual = |teacher set| at final state.
# Tree sets from canonical st[t][0].observation.players[slot][3] keys (same source cnet2_build uses).
# CPU-only, read-only. Marker: M9XJPROF2012A2
MARK=M9XJPROF2012A2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M9XJPROF2012A2 DGX-BEGIN $(date '+%F %T %Z')"
cd /home/arm/junyang/carbon-repo || { echo NO-REPO; exit 1; }
echo "HEAD $(git rev-parse HEAD)"
python3 - <<'PYEOF'
import json, gzip, glob, os, math
from statistics import mean, median, pstdev

TEACHER = "RyanX-AIPowercore-M9XJ-it260"

def pct(vals, q):
    if not vals: return None
    s = sorted(vals); k = (len(s) - 1) * q / 100.0
    f = math.floor(k); c = math.ceil(k)
    if f == c: return s[int(k)]
    return s[int(f)] * (c - k) + s[int(c)] * (k - f)

def desc(vals):
    if not vals: return {}
    return {"n": len(vals), "mean": round(mean(vals), 2), "median": round(median(vals), 2),
            "sd": round(pstdev(vals), 2) if len(vals) > 1 else 0.0,
            "p25": round(pct(vals, 25), 1), "p75": round(pct(vals, 75), 1),
            "min": min(vals), "max": max(vals)}

matches = json.load(open('data/m9xj_matches.json'))
by_id = {e["id"]: e for e in matches}
files = sorted(glob.glob('data/m9xj_corpus/*.json.gz'))
print("matches entries:", len(matches), "corpus files:", len(files))

rows = []
unmatched_files = []
reward_sign_ok = reward_sign_bad = reward_sign_skip = 0
traj_printed = 0
obs_trees_info = None
for fp in files:
    stem = os.path.basename(fp)[:-len(".json.gz")]
    gid, _, gnum = stem.rpartition("-")
    e = by_id.get(gid)
    if e is None:
        unmatched_files.append(stem); continue
    tkey = "a" if e["name_a"] == TEACHER else ("b" if e["name_b"] == TEACHER else None)
    if tkey is None:
        unmatched_files.append(stem + ":noteacher"); continue
    slot = list(e["order"]).index(tkey)
    d = json.load(gzip.open(fp))
    st = d["steps"]
    def tset(t, s):
        pl = st[t][0]["observation"]["players"][s]
        c = pl[3]
        return set(c.keys()) if isinstance(c, dict) else set()
    if obs_trees_info is None:
        mid = st[len(st)//2][0]["observation"]
        ot = mid.get("trees")
        info = {"type": type(ot).__name__}
        if isinstance(ot, dict):
            info["len"] = len(ot)
            info["sample"] = list(ot.items())[:2]
        obs_trees_info = info
        p3 = mid["players"][slot][3]
        if isinstance(p3, dict) and p3:
            print("players[slot][3] sample items:", list(p3.items())[:2])
        print("obs['trees'] info:", json.dumps(obs_trees_info, ensure_ascii=False, default=str)[:500])
    A = [tset(t, slot) for t in range(len(st))]
    B = [tset(t, 1 - slot) for t in range(len(st))]
    plants = seized = plants_opp = 0
    for t in range(len(st) - 1):
        newA = A[t + 1] - A[t]
        plants += len(newA - B[t])
        seized += len(newA & B[t])
        newB = B[t + 1] - B[t]
        plants_opp += len(newB - A[t])
    sc = e["scores"]
    tscore = sc[tkey]; oscore = sc["b" if tkey == "a" else "a"]
    margin = round(tscore - oscore, 1)
    won = e.get("winner") == tkey
    # slot-mapping cross-check vs replay rewards (sign agreement)
    rw = d.get("rewards")
    if isinstance(rw, (list, tuple)) and len(rw) == 2 and all(isinstance(x, (int, float)) for x in rw):
        rs = (rw[slot] > rw[1 - slot]) - (rw[slot] < rw[1 - slot])
        ms = (margin > 0) - (margin < 0)
        if rs == 0 or ms == 0: reward_sign_skip += 1
        elif rs == ms: reward_sign_ok += 1
        else: reward_sign_bad += 1
    else:
        reward_sign_skip += 1
    if traj_printed < 3:
        traj = [(t, len(A[t]), len(B[t])) for t in range(0, len(st), 25)] + [(len(st) - 1, len(A[-1]), len(B[-1]))]
        print("TRAJ", stem[:8], "slot", slot, traj)
        print("  statuses:", json.dumps(d.get("statuses"), default=str)[:160], "rewards:", json.dumps(d.get("rewards"), default=str)[:120])
        traj_printed += 1
    rows.append({"id": gid[:8], "slot": slot, "steps": len(st), "plants": plants, "seized": seized,
                 "plants_opp": plants_opp, "residual": len(A[-1]), "residual_opp": len(B[-1]),
                 "treesteps": sum(len(a) for a in A), "init_trees": len(A[0]) + len(B[0]),
                 "margin": margin, "won": won})

print("analyzed:", len(rows), "unmatched files:", unmatched_files)
print("reward-sign check ok/bad/skip:", reward_sign_ok, reward_sign_bad, reward_sign_skip)
print("games with nonzero initial trees:", sum(1 for r in rows if r["init_trees"] > 0))
out = {}
out["games"] = len(rows)
out["plants_all"] = desc([r["plants"] for r in rows])
out["residual_all"] = desc([r["residual"] for r in rows])
out["seized_all"] = desc([r["seized"] for r in rows])
out["treesteps_all"] = desc([r["treesteps"] for r in rows])
out["plants_opp_all"] = desc([r["plants_opp"] for r in rows])
out["residual_opp_all"] = desc([r["residual_opp"] for r in rows])
for s in (0, 1):
    sub = [r for r in rows if r["slot"] == s]
    out[f"slot{s}"] = {"n": len(sub), "plants": desc([r["plants"] for r in sub]),
                       "residual": desc([r["residual"] for r in sub]),
                       "seized": desc([r["seized"] for r in sub]),
                       "winrate": round(sum(1 for r in sub if r["won"]) / len(sub), 3) if sub else None}
w = [r for r in rows if r["won"]]; l = [r for r in rows if not r["won"]]
out["teacher_record"] = {"W": len(w), "L": len(l)}
out["plants_when_won"] = desc([r["plants"] for r in w])
out["plants_when_lost"] = desc([r["plants"] for r in l])
out["residual_when_won"] = desc([r["residual"] for r in w])
out["residual_when_lost"] = desc([r["residual"] for r in l])
out["margin_all"] = desc([r["margin"] for r in rows])
print("RESULT-JSON-BEGIN")
print(json.dumps(out, ensure_ascii=False))
print("RESULT-JSON-END")
print("PERGAME-BEGIN")
for r in rows:
    print(json.dumps(r, ensure_ascii=False))
print("PERGAME-END")
PYEOF
echo "M9XJPROF2012A2 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
