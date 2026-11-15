#!/bin/bash
# m6_corpus closure verification (war-room agent; read-only).
# The 18:34 M6SYNC1833ZQ4 mirror run (rsync --delete, run 43201498135)
# already removed the stale index.json + replays/; this probe independently
# confirms the end state and md5-compares workdir gz files against the repo.
# Must NOT touch training processes or any weights.
# Marker: M6VRF1846K2
MARK=M6VRF1846K2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M6VRF1846K2 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- TRAIN-CHECK ---'
ps -eo pid,etime,args | grep 'cnet2_train.py.*m7bsv2' | grep -v grep || echo NO-M7BSV2-TRAIN
echo '--- REPO-PULL ---'
cd /home/arm/junyang/carbon-repo
timeout 60 git pull --ff-only 2>&1 | tail -2
echo "repo HEAD: $(git rev-parse HEAD) $(git log -1 --format='%ci %s')"
echo '--- REPO-COUNTS ---'
echo -n "repo m6_corpus gz: "; ls data/m6_corpus/*.json.gz 2>/dev/null | wc -l
echo -n "repo m6_matches.json entries: "; python3 -c "import json;print(len(json.load(open('data/m6_matches.json'))))" 2>/dev/null || echo NO-MATCHES
ls data/m6_corpus/index.json 2>/dev/null && echo REPO-HAS-INDEX || echo REPO-NO-INDEX
echo '--- WORKDIR-COUNTS ---'
WORK_DIR=/home/arm/junyang/carbon/data/m6_corpus
echo -n "workdir top-level gz: "; ls "$WORK_DIR"/*.json.gz 2>/dev/null | wc -l
echo -n "workdir total files: "; find "$WORK_DIR" -type f 2>/dev/null | wc -l
ls "$WORK_DIR/index.json" 2>/dev/null && echo WORK-HAS-INDEX || echo WORK-NO-INDEX
ls -d "$WORK_DIR/replays" 2>/dev/null && echo WORK-HAS-REPLAYS-DIR || echo WORK-NO-REPLAYS-DIR
echo '--- MD5-COMPARE (repo gz vs workdir gz, by filename) ---'
REPO_DIR=/home/arm/junyang/carbon-repo/data/m6_corpus
(cd "$REPO_DIR" && md5sum *.json.gz | sort -k2) > /tmp/m6v_repo_md5.txt
(cd "$WORK_DIR" && md5sum *.json.gz | sort -k2) > /tmp/m6v_work_md5.txt
if diff -q /tmp/m6v_repo_md5.txt /tmp/m6v_work_md5.txt >/dev/null 2>&1; then echo MD5-MATCH-ALL; else echo MD5-DIFF; diff /tmp/m6v_repo_md5.txt /tmp/m6v_work_md5.txt | head -10; fi
echo "M6VRF1846K2 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
