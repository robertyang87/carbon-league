#!/bin/bash
# DGX status probe trip 13:28 deploy re-check - read-only. Marker DGXSTATUS1328C.
echo "DGXSTATUS1328C $(date '+%H:%M:%S %Z')"
ps -eo pid,etime,cmd | grep -i deploy | grep -v grep || echo NO_DEPLOY_PROC
ls -lt /home/arm/junyang/golden/ 2>&1 | head -5
find /home/arm/junyang -maxdepth 2 -name '*EXAM*' -o -maxdepth 2 -name '*READY*' 2>/dev/null | head -5
ls -lt /home/arm/junyang/ 2>/dev/null | head -12
echo "DGXSTATUS1328C-END $(date '+%H:%M:%S %Z')"
