#!/bin/bash
# DGX status probe 22b (follow-up, same run): fill two gaps from 17:59 read:
# ds_m7bs_v2 meta sample count after 17:53 rebuild; build log tail; current epoch.
# Marker: DGXSTATUS1758B2K8
MARK=DGXSTATUS1758B2K8
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1758B2K8 DGX-BEGIN $(date '+%F %T %Z')"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
ps -eo pid,etime,args | grep -E 'cnet2_train' | grep -v grep || echo NO-TRAIN-PROC
echo '--- DS_M7BS_V2-META ---'
python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/ds_m7bs_v2/meta.json'));print({k:d.get(k) for k in ('n_train','n_val','n','samples','games','teacher','num_games') if k in d});print('keys:',sorted(d.keys())[:20])" 2>&1
echo '--- BUILD-LOG-TAIL ---'
tail -6 /home/arm/junyang/carbon/data/dq_logs/build_m7bsv2.log 2>/dev/null || echo NO-BUILD-LOG
echo '--- M7V2FIX-LOG-TAIL ---'
tail -6 /home/arm/junyang/carbon/data/dq_logs/m7v2fix.log 2>/dev/null || echo NO-FIX-LOG
echo '--- TRAIN-LOG-TAILS ---'
for f in /home/arm/junyang/carbon/data/dq_logs/m7bsv2r1.log /home/arm/junyang/carbon/data/dq_logs/m7bsv2g1.log; do echo "== $f"; tail -2 "$f" 2>/dev/null; done
echo "DGXSTATUS1758B2K8 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
