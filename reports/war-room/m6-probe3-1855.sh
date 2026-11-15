#!/bin/bash
# M6 probe 3: how K4v2 built its index (stage log), ladder_corpus sync/fetch heads,
# and the pre-existing ds_m6 directory.
# Marker: M6PRB31855K3
MARK=M6PRB31855K3
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M6PRB31855K3 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- K4V2-STAGE-LOG ---'
cat /home/arm/junyang/carbon/data/dq_logs/k4v2_stage.log 2>&1
echo '--- M7V2FIX-LOG ---'
cat /home/arm/junyang/carbon/data/dq_logs/m7v2fix.log 2>&1
echo '--- LADDER-CORPUS-HEAD ---'
sed -n '1,110p' /home/arm/junyang/carbon/code/ladder_corpus.py 2>&1
echo '--- LADDER-COOKIE-PATH ---'
grep -n "cookie" /home/arm/junyang/carbon/code/ladder_corpus.py /home/arm/junyang/carbon/code/ladder.py 2>&1 | head -10
ls /home/arm/junyang/carbon/code/ | grep -i ladder
echo '--- DS-M6-EXISTING ---'
ls -la /home/arm/junyang/carbon/data/ds_m6/ 2>&1 | head -20
cat /home/arm/junyang/carbon/data/ds_m6/meta.json 2>&1 | head -5
echo '--- M6-REPLAY-SAMPLE ---'
/home/arm/junyang/carbon/.venv/bin/python -c "
import gzip, json
f='/home/arm/junyang/carbon/data/m6_corpus/013a4640b677df4d577848165387f600-1.json.gz'
d=json.load(gzip.open(f,'rt'))
print('replay-type', type(d).__name__)
print('replay-keys', sorted(d.keys())[:20] if isinstance(d,dict) else len(d))
" 2>&1
echo "M6PRB31855K3 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
