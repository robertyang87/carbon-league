#!/bin/bash
# DGX status probe trip 13:28 spot-31 manifest verify - read-only. Marker DGXSTATUS1328E.
echo "DGXSTATUS1328E $(date '+%H:%M:%S %Z')"
KEY="$HOME/.ssh/id_ed25519_carbon_relay"
SSH_OPTS="-i $KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
timeout 60 ssh $SSH_OPTS root@100.98.67.20 'hostname; /root/carbon/.venv/bin/python -c "import numpy; print(\"numpy\", numpy.__version__)"; md5sum /root/carbon/code/eval_v53.py /root/carbon/code/bigeval.py /root/carbon/code/cnet2_search.py; ls -d /root/carbon/data/weights; ls -l /root/exam_stack_golden.tar.zst 2>/dev/null' 2>&1
echo "rc=$?"
echo "DGXSTATUS1328E-END $(date '+%H:%M:%S %Z')"
