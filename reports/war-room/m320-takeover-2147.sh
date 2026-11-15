#!/bin/bash
# it320 chain TAKEOVER probe (war-room exec agent, 2026-11-12 ~21:5x).
# Previous chain agent left a WAIT-FREEZE poller (PID 264410 on DGX, alive 21:43).
# Freeze conditions met 21:47 (91 games, corpus diff 0, Echo2=6). Poller may or may
# not have self-triggered the chain. This probe is READ-ONLY: establish ground truth
# before any action (anti double-execution).
# Marker: M320TAKEOVER2147 (claim only output containing this marker)
MARK=M320TAKEOVER2147
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M320TAKEOVER2147 DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo '--- POLLER-264410 ---'
if [ -d /proc/264410 ]; then ps -p 264410 -o pid,etime,args; tr '\0' ' ' < /proc/264410/cmdline; echo; echo "cwd=$(readlink /proc/264410/cwd)"; else echo POLLER-264410-GONE; fi
echo '--- CHAIN-RELATED-PROCS ---'
ps -eo pid,etime,args | grep -E 'replica_chain|build_top|build_m9xj|cnet2_train|m9xj320|WAIT|freeze|poller|set_uw' | grep -v grep || echo NO-CHAIN-PROC
echo '--- ALL-PYTHON ---'
ps -eo pid,etime,args | grep -E 'python' | grep -v grep | head -12 || echo NO-PYTHON
echo '--- GIT-LEFTOVER ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NO-GIT-PROC
echo '--- REPO-HEAD ---'
(cd /home/arm/junyang/carbon-repo && git rev-parse HEAD && git branch --show-current && echo -n 'remote: ' && timeout 60 git ls-remote origin cnet2-search-v12) 2>&1
echo '--- WORKDIR-CORPUS ---'
ls /home/arm/junyang/carbon/data/m9xj320_corpus/ 2>/dev/null | wc -l
python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon-repo/data/m9xj320_matches.json'));print('repo_matches_entries=',len(d))" 2>&1
echo '--- DS-DIRS ---'
for d in ds_m9xj320 ds_m9xj320_w ds_m9xj320_85scrapped; do echo "== $d"; ls -ld --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/$d 2>/dev/null || echo ABSENT; ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/$d/meta.json 2>/dev/null; python3 -c "import json;m=json.load(open('/home/arm/junyang/carbon/data/$d/meta.json'));print({k:(len(v) if isinstance(v,list) else v) for k,v in m.items()})" 2>/dev/null; done
echo '--- READY-MANIFEST ---'
ls -lt --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/ | head -25
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/*READY* /home/arm/junyang/carbon/data/*manifest* /home/arm/junyang/carbon/data/*m9xj320* 2>/dev/null | head -20
echo '--- CHAIN-STATE-FILES ---'
find /home/arm/junyang/carbon -maxdepth 3 -name '*chain*state*' -o -maxdepth 3 -name '*m320*state*' 2>/dev/null | head; ls -lt --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/dq_logs/ 2>/dev/null | head -15
echo '--- M320-LOGS-TAIL ---'
for f in /home/arm/junyang/carbon/data/dq_logs/*m9xj320* /home/arm/junyang/carbon/data/dq_logs/*m320* /home/arm/junyang/carbon/data/*m9xj320*.log; do [ -f "$f" ] && { echo "== $f"; tail -6 "$f"; }; done; echo logs-tail-done
echo '--- WEIGHTS-M320 ---'
ls -lt --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/weights/ 2>/dev/null | head -8
echo '--- RECENT-UNDER-DATA-60MIN ---'
find /home/arm/junyang/carbon/data -mmin -60 -type f 2>/dev/null | grep -v m9xj320_corpus | head -30; echo recent-done
echo "M320TAKEOVER2147 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
