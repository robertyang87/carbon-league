#!/bin/bash
# M6 verification probe (read-only): stage-log outcome + live training trio.
# Marker: M6VRF1913Q7
MARK=M6VRF1913Q7
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M6VRF1913Q7 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- STAGE-OUTCOME ---'
grep -h "M6STG1 \(LAUNCH-OK\|FAIL\|STAGE-BEGIN\)" /home/arm/junyang/carbon/data/dq_logs/m6_stage.log 2>/dev/null || echo NO-STAGE-VERDICT-YET
grep -h "THREE-COUNT\|scan games:\|DS-META games:\|INDEX-BUILT\|build_m6_rc=\|set_uw_rc=\|train pids:\|ep0 lines:\|repo matches=" /home/arm/junyang/carbon/data/dq_logs/m6_stage.log 2>/dev/null
echo '--- STAGE-TAIL ---'
tail -25 /home/arm/junyang/carbon/data/dq_logs/m6_stage.log 2>/dev/null | cut -c1-300
echo '--- STAGE-PROC ---'
pgrep -af 'm6-dgx-stage' | grep -v grep || echo STAGE-SCRIPT-GONE
echo '--- TRAIN-TRIO ---'
ps -eo pid,etime,args | grep 'cnet2_train' | grep -v grep || echo NO-TRAIN-PROC
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
ls -l --time-style=+%F_%T /home/arm/junyang/carbon/data/dq_logs/m6r1.log /home/arm/junyang/carbon/data/dq_logs/m6g1.log 2>&1
echo "ep0-r1-lines: $(grep -c '\"ep\": 0' /home/arm/junyang/carbon/data/dq_logs/m6r1.log 2>/dev/null || echo 0)"
echo "ep0-g1-lines: $(grep -c '\"ep\": 0' /home/arm/junyang/carbon/data/dq_logs/m6g1.log 2>/dev/null || echo 0)"
echo "ep0-r1-val: $(grep -h '\"ep\": 0' /home/arm/junyang/carbon/data/dq_logs/m6r1.log 2>/dev/null | tail -1 | cut -c1-300)"
echo "ep0-g1-val: $(grep -h '\"ep\": 0' /home/arm/junyang/carbon/data/dq_logs/m6g1.log 2>/dev/null | tail -1 | cut -c1-300)"
echo '--- DS-META ---'
/home/arm/junyang/carbon/.venv/bin/python -c 'import json,sys; d=json.load(open(sys.argv[1])); g=d.get("games"); print("DS-META games:", len(g) if isinstance(g, list) else g, "total:", d.get("total"))' /home/arm/junyang/carbon/data/ds_m6/meta.json 2>&1
ls -l --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/cnet2_m6r1*.npz /home/arm/junyang/carbon/data/weights/cnet2_m6g1*.npz 2>&1 | head -5
echo "M6VRF1913Q7 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
