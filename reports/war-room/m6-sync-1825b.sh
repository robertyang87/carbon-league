#!/bin/bash
# Follow-up read-only check after m6 sync: file-format breakdown + repo m6 files + g1 completion.
# Marker: M6SYNC1825B2Q9
MARK=M6SYNC1825B2Q9
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M6SYNC1825B2Q9 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- REPO-M6-FILES ---'
cd /home/arm/junyang/carbon-repo
find data -maxdepth 2 -iname '*m6*' | sort
echo '--- WORKDIR-EXT-BREAKDOWN ---'
ls /home/arm/junyang/carbon/data/m6_corpus/ | sed 's/.*\.//' | sort | uniq -c
echo -n "workdir .json count: "; ls /home/arm/junyang/carbon/data/m6_corpus/*.json 2>/dev/null | wc -l
echo -n "workdir .json.gz count: "; ls /home/arm/junyang/carbon/data/m6_corpus/*.json.gz 2>/dev/null | wc -l
echo '--- SAMPLE-OLD-FILE ---'
ls /home/arm/junyang/carbon/data/m6_corpus/*.json 2>/dev/null | head -2
echo '--- G1-COMPLETION ---'
ps -eo pid,etime,args | grep 'cnet2_train.py.*m7bsv2' | grep -v grep || echo NO-M7BSV2-TRAIN
tail -2 /home/arm/junyang/carbon/data/dq_logs/m7bsv2g1.log
tail -2 /home/arm/junyang/carbon/data/dq_logs/m7bsv2r1.log
ls -lt --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/weights/ | grep 'm7bsv2' | head -8
ls /home/arm/junyang/carbon/data/dq_logs/*.done 2>/dev/null | tail -5
echo "M6SYNC1825B2Q9 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
