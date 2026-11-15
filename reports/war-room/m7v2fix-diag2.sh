#!/bin/bash
# M7V2 fix DIAG2: DGX state read via L20->DGX ssh (same wrapper as the
# working probe template). Marker: __DIAG_MARK__
MARK=__DIAG_MARK__
echo "$MARK DIAG-BEGIN $(date '+%F %T %Z')"
timeout 120 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'echo DGX-HOST $(hostname) $(date "+%F %T %Z"); echo --- stage-proc ---; pgrep -af "m7v2fix" || echo NO-STAGE-PROC; echo --- train-procs ---; pgrep -af "cnet2_train[.]py" || echo NO-TRAIN-PROC; echo --- stage-log ---; wc -l /home/arm/junyang/carbon/data/dq_logs/m7v2fix.log; stat -c "%y" /home/arm/junyang/carbon/data/dq_logs/m7v2fix.log; tail -n 8 /home/arm/junyang/carbon/data/dq_logs/m7v2fix.log | cut -c1-200; echo --- build-log-tail ---; tail -n 3 /home/arm/junyang/carbon/data/dq_logs/build_m7bsv2.log | cut -c1-300; echo --- meta ---; cat /home/arm/junyang/carbon/data/ds_m7bs_v2/meta.json 2>&1 | cut -c1-300; echo --- trainer-files ---; ls -la /home/arm/junyang/carbon/data/dq_logs/m7bsv2r1.log /home/arm/junyang/carbon/data/dq_logs/m7bsv2g1.log /home/arm/junyang/carbon/data/weights/cnet2_m7bsv2r1_ep19.npz /home/arm/junyang/carbon/data/weights/cnet2_m7bsv2g1_ep19.npz 2>&1; echo --- gpu ---; nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used --format=csv,noheader'
echo "$MARK DIAG-END"
