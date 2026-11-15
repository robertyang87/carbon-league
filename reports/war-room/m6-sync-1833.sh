#!/bin/bash
# m6_corpus clean mirror redo (war-room 18:32 re-instruction; executor: DGX reporter).
# Repo authoritative. Mirror = rsync --delete ONLY if every old workdir stem exists in repo.
# Must NOT touch running m7bsv2 training processes.
# Marker: M6SYNC1833ZQ4
MARK=M6SYNC1833ZQ4
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M6SYNC1833ZQ4 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- TRAIN-CHECK (before) ---'
ps -eo pid,etime,args | grep 'cnet2_train.py.*m7bsv2' | grep -v grep || echo NO-M7BSV2-TRAIN
echo '--- REPO-PULL ---'
cd /home/arm/junyang/carbon-repo
timeout 60 git pull --ff-only 2>&1 | tail -3
echo "repo HEAD: $(git rev-parse HEAD) $(git log -1 --format='%ci %s')"
echo '--- REPO-COUNTS ---'
echo -n "repo m6_corpus gz: "; ls data/m6_corpus/*.json.gz 2>/dev/null | wc -l
echo -n "repo m6_matches.json entries: "; python3 -c "import json;print(len(json.load(open('data/m6_matches.json'))))" 2>/dev/null || echo NO-MATCHES
ls data/m6_corpus/index.json 2>/dev/null && echo REPO-HAS-INDEX || echo REPO-NO-INDEX
echo '--- OVERLAP-CHECK (old workdir stems vs repo) ---'
REPO_DIR=/home/arm/junyang/carbon-repo/data/m6_corpus
WORK_DIR=/home/arm/junyang/carbon/data/m6_corpus
ls "$REPO_DIR"/*.json.gz | xargs -n1 basename | sed 's/\.json\.gz$//' | sort > /tmp/m6_repo_stems.txt
find "$WORK_DIR/replays" -type f 2>/dev/null | xargs -n1 basename 2>/dev/null | sed -e 's/\.json\.gz$//' -e 's/\.json$//' | sort -u > /tmp/m6_old_stems.txt
echo "old stems: $(wc -l < /tmp/m6_old_stems.txt)"
MISSING=$(comm -23 /tmp/m6_old_stems.txt /tmp/m6_repo_stems.txt | wc -l)
echo "old stems missing from repo: $MISSING"
if [ "$MISSING" -eq 0 ]; then
  echo '--- MIRROR (rsync --delete, repo authoritative) ---'
  rsync -a --delete "$REPO_DIR/" "$WORK_DIR/" 2>&1 | tail -2
  echo "mirror rc=$?"
else
  echo "SKIP-MIRROR missing=$MISSING"
  comm -23 /tmp/m6_old_stems.txt /tmp/m6_repo_stems.txt | head -5
fi
echo '--- WORKDIR-COUNTS-AFTER ---'
echo -n "workdir m6_corpus gz after: "; ls "$WORK_DIR"/*.json.gz 2>/dev/null | wc -l
echo -n "workdir m6_corpus total files after: "; find "$WORK_DIR" -type f | wc -l
ls "$WORK_DIR/index.json" 2>/dev/null && echo WORK-HAS-INDEX || echo WORK-NO-INDEX
echo '--- TRAIN-CHECK (after) ---'
ps -eo pid,etime,args | grep 'cnet2_train.py.*m7bsv2' | grep -v grep || echo NO-M7BSV2-TRAIN-AFTER
tail -1 /home/arm/junyang/carbon/data/dq_logs/m7bsv2r1.log
echo "M6SYNC1833ZQ4 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
