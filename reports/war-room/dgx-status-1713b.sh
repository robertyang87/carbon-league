#!/bin/bash
# DGX status probe 19b (follow-up): first probe found NEW file m7v2.log (17:10, 1401B) + fresh m7bs_corpus replays,
# but build_m7bs_v2.log / ds_m7bs_v2 / m7bsv2 train logs all missing. Read the actual M7Bs v2 progress log.
# Marker: DGXSTATUS1713B2K8 (claim only output containing this marker)
MARK=DGXSTATUS1713B2K8
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1713B2K8 DGX-BEGIN $(date '+%F %T %Z')"
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader
ps -eo pid,etime,args | grep -E 'cnet2_train|m7bs|m7v2|train|carago|fable|build_' | grep -v grep || echo NO-TRAIN-PROC
echo '== m7v2.log FULL =='
cat /home/arm/junyang/carbon/data/dq_logs/m7v2.log 2>/dev/null || echo NO-M7V2-LOG
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/dq_logs/m7v2.log
echo '== m7bs_corpus counts =='
echo -n 'replays='; ls /home/arm/junyang/carbon/data/m7bs_corpus/replays/ 2>/dev/null | wc -l
python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/m7bs_corpus/index.json'));print('index_entries=',len(d if isinstance(d,list) else d.get('entries') or d))" 2>&1
echo -n 'replays mtime<30min: '; find /home/arm/junyang/carbon/data/m7bs_corpus/replays -mmin -30 -type f 2>/dev/null | wc -l
echo -n 'replays mtime<120min: '; find /home/arm/junyang/carbon/data/m7bs_corpus/replays -mmin -120 -type f 2>/dev/null | wc -l
echo '== ds_m7bs_v2 =='
ls -la /home/arm/junyang/carbon/data/ds_m7bs_v2 2>/dev/null || echo NO-DS-M7BS-V2-DIR
echo '== m7bs_v2 build/train logs anywhere =='
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/dq_logs/ | grep -i 'm7' ; find /home/arm/junyang/carbon/data -maxdepth 2 -name '*m7v2*' -o -maxdepth 2 -name '*m7bs_v2*' 2>/dev/null | head
echo '== any python/build proc etime =='
ps -eo pid,etime,args | grep -E 'python|build' | grep -v grep | head -8 || echo NO-PYTHON-BUILD-PROC
echo "DGXSTATUS1713B2K8 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
