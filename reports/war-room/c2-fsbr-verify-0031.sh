echo C2FSBR0031B_BEGIN
echo "--- backup fastsim md5 (expect old 61859708) ---"
md5sum /root/carbon/code_backup_pre_fastsim_20261114/fastsim.py 2>&1
ls -l /root/carbon/code_backup_pre_fastsim_20261114/ 2>&1
echo "--- newsim live stats ---"
python3 - <<'EOF'
import json, math
def stats(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = len(rows)
    print("KEYS:", sorted(rows[0].keys()))
    def mg(r):
        for k in ("margin", "m", "score", "delta"):
            if k in r: return r[k]
        return None
    ms = [mg(r) for r in rows]
    assert all(m is not None for m in ms), "margin key not found"
    wins = sum(1 for m in ms if m > 0)
    mean = sum(ms)/n
    var = sum((m-mean)**2 for m in ms)/(n-1)
    se = math.sqrt(var/n)
    seeds = [r.get("seed", r.get("seed0")) for r in rows]
    errs = sum(1 for r in rows if r.get("err"))
    print(path, "n=", n, "wins=", wins, "mean=", round(mean,1), "se=", round(se,1), "lo2se=", round(mean-2*se,1), "worst=", min(ms), "err=", errs, "seed_first=", seeds[0], "seed_last=", seeds[-1])
stats("/root/carbon/data/bigeval_live/snap_70_echo2_newsim.jsonl")
stats("/root/carbon/data/bigeval_live/snap_70_echo2.jsonl")
EOF
echo "--- smoke ---"
cat /root/carbon/data/bigeval_live/snap_70_ab_smoke.jsonl 2>/dev/null | head -5
echo "--- newsim launcher log ---"
tail -25 /root/carbon/data/snap_70_echo2_newsim.launcher.log 2>&1
echo "--- ab smoke launcher log ---"
ls -l /root/carbon/data/*ab_smoke* /root/carbon/data/*newsim* 2>&1
echo C2FSBR0031B_END
