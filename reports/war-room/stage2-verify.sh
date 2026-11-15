#!/bin/bash
# War-room verify for staging stage2: stage2.log + build logs + training three-piece.
# Marker: VFYSTAGE2R5
MARK=VFYSTAGE2R5
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "VFYSTAGE2R5 DGX-BEGIN $(date '+%F %T %Z')"
D=/home/arm/junyang/carbon/data
echo '--- STAGE2-LOG ---'
tail -50 $D/dq_logs/stage2.log
echo '--- BUILD-CARAGO-SUMMARY ---'
grep -E 'games|samples|teacher' $D/dq_logs/build_carago.log | head -8
echo '--- BUILD-FABLE-SUMMARY ---'
grep -E 'games|samples|teacher' $D/dq_logs/build_fable.log | head -8
echo '--- TRAIN-THREE-PIECE ---'
ps -eo pid,etime,args | grep -E 'cnet2_train' | grep -v grep || echo NO-TRAIN-PROC
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader
echo '--- CARAGO-TRAIN-LOGS ---'
tail -3 $D/dq_logs/caragor1.log 2>/dev/null
tail -3 $D/dq_logs/caragog1.log 2>/dev/null
echo '--- FABLE-DS ---'
ls $D/ds_fable/meta.json 2>/dev/null && du -sh $D/ds_fable
echo "VFYSTAGE2R5 DGX-END"
DGXEOF
echo "$MARK SSH_RC=$? L20-END"
