#!/bin/bash
# Verify current m6_corpus state after 18:34 mirror (war-room 18:42 method: repo=191 flat, workdir must be 191 gz, residue 0).
# Read-only verification. Must NOT touch training (none expected running).
# Marker: M6SYNC1843CHK
MARK=M6SYNC1843CHK
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M6SYNC1843CHK DGX-BEGIN $(date '+%F %T %Z')"
echo '--- REPO ---'
cd /home/arm/junyang/carbon-repo
echo "repo HEAD: $(git rev-parse --short HEAD) $(git log -1 --format='%ci %s')"
echo -n "repo m6_corpus gz: "; ls data/m6_corpus/*.json.gz 2>/dev/null | wc -l
echo -n "repo m6_matches.json entries: "; python3 -c "import json;print(len(json.load(open('data/m6_matches.json'))))" 2>/dev/null || echo NO-MATCHES
ls data/m6_corpus/index.json 2>/dev/null && echo REPO-HAS-INDEX || echo REPO-NO-INDEX
echo '--- WORKDIR-M6 ---'
W=/home/arm/junyang/carbon/data/m6_corpus
echo -n "workdir top gz: "; ls "$W"/*.json.gz 2>/dev/null | wc -l
echo -n "workdir total files: "; find "$W" -type f | wc -l
echo -n "workdir subdirs: "; find "$W" -mindepth 1 -type d | wc -l
ls "$W/index.json" 2>/dev/null && echo WORK-HAS-INDEX || echo WORK-NO-INDEX
ls "$W/replays" 2>/dev/null && echo WORK-HAS-REPLAYS-SUBDIR || echo WORK-NO-REPLAYS-SUBDIR
echo '--- M7BSV2-FINAL ---'
ps -eo pid,etime,args | grep 'cnet2_train.py.*m7bsv2' | grep -v grep || echo NO-M7BSV2-TRAIN
tail -1 /home/arm/junyang/carbon/data/dq_logs/m7bsv2r1.log
tail -1 /home/arm/junyang/carbon/data/dq_logs/m7bsv2g1.log
ls -lt --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/weights/ | grep 'm7bsv2.*ep19' | head -4
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader
echo "M6SYNC1843CHK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
