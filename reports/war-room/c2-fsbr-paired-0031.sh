echo C2FSBR0031C_BEGIN
python3 - <<'EOF'
import json, math
def load(p):
    d = {}
    for l in open(p):
        if not l.strip(): continue
        r = json.loads(l)
        d[r["seed"]] = r["margin"]
    return d
old = load("/root/carbon/data/bigeval_live/snap_70_echo2.jsonl")
new = load("/root/carbon/data/bigeval_live/snap_70_echo2_newsim.jsonl")
print("old_n=", len(old), "new_n=", len(new), "seed_sets_equal=", set(old) == set(new))
print("seed_range_old=", min(old), max(old), "seed_range_new=", min(new), max(new))
diffs = [new[s] - old[s] for s in sorted(old) if s in new]
ndiff = sum(1 for d in diffs if d != 0)
flips = sum(1 for s in sorted(old) if s in new and (new[s] > 0) != (old[s] > 0))
print("games_compared=", len(diffs), "differing=", ndiff, "flips=", flips)
if diffs:
    m = sum(diffs)/len(diffs)
    var = sum((d-m)**2 for d in diffs)/(len(diffs)-1) if len(diffs) > 1 else 0.0
    se = math.sqrt(var/len(diffs))
    print("paired_margin_delta mean=", round(m,3), "se=", round(se,3), "95CI=[", round(m-1.96*se,3), ",", round(m+1.96*se,3), "] max|d|=", max(abs(d) for d in diffs))
    print("identical_per_game=", all(d == 0 for d in diffs))
EOF
echo "--- snapshot loop on machine? ---"
ps -eo pid,etime,args | grep -iE "snap|sync|gh_api" | grep -v grep || echo NO-SNAP-PROC-ON-MACHINE
echo C2FSBR0031C_END
