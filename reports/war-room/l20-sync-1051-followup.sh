#!/bin/bash
# War-room-sync v131 follow-up: read it20 checkpoint field RESULTS off L20. READ-ONLY. Marker L20SYNC1051B.
# Context: 10:53:58 probe showed E2 field already running (PIDs 219650+, etime 01:49) => M2 field
# (seeds 980000-981999, 2000 games) finished ~10:52. Live files live under /root/carbon/probe_m2/data/bigeval_live/.
MARK=L20SYNC1051B
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/war-room-sync-1051-l20followup.log
{
echo "$MARK BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'bash -s' <<'STEOF'
D=/root/carbon/probe_m2/data/bigeval_live
echo '--- M2-FIELD ---'
wc -l $D/ppou2_m_it20_m2.jsonl 2>&1
tail -2 $D/ppou2_m_it20_m2.jsonl 2>&1 | cut -c1-300
echo '--- M2-RESULT-COMPUTE ---'
python3 - "$D/ppou2_m_it20_m2.jsonl" <<'PYEOF'
import json, sys, math
p = sys.argv[1]
rows = [json.loads(l) for l in open(p) if l.strip()]
n = len(rows)
errs = sum(1 for r in rows if r.get("err"))
wins = sum(1 for r in rows if r.get("margin", 0) > 0)
ms = [r["margin"] for r in rows if "margin" in r]
mean = sum(ms)/len(ms) if ms else float("nan")
sd = (sum((x-mean)**2 for x in ms)/(len(ms)-1))**0.5 if len(ms) > 1 else 0.0
se = sd/math.sqrt(len(ms)) if ms else 0.0
seeds = sorted(r["seed"] for r in rows)
print(f"M2 n={n} err={errs} wins={wins} losses={n-wins-errs} mean={mean:.1f} se={se:.1f} lo2se={mean-2*se:.1f} seed_range={seeds[0]}-{seeds[-1]} contiguous={seeds==list(range(seeds[0], seeds[0]+n))}")
PYEOF
echo '--- M2-LOG-TAIL ---'
tail -15 /root/carbon/probe_m2/it20_m2.log 2>&1 | cut -c1-300
echo '--- E2-FIELD ---'
wc -l $D/ppou2_m_it20_e2.jsonl 2>&1
tail -1 $D/ppou2_m_it20_e2.jsonl 2>&1 | cut -c1-300
echo '--- E2-PROCS ---'
ps -eo pid,etime,args | grep -c '[b]igeval.py exp/ppou2_m_it20.py exp/echo2.py'
echo '--- DONE-FILE ---'
ls -l --time-style=+%F_%T /root/carbon/probe_m2/it20_done.txt 2>&1
echo "L20SYNC1051B L20-END $(date '+%F %T %Z')"
STEOF
echo "$MARK END $(date '+%F %T %Z')"
} > "$LOG" 2>&1
echo "exit=$?"
cat "$LOG"
