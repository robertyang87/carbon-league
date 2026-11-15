#!/bin/bash
# DGX status probe trip 13:28 deploy post-mortem - read-only. Marker DGXSTATUS1328D.
echo "DGXSTATUS1328D $(date '+%H:%M:%S %Z')"
echo "== deploy_stack.sh =="
cat /home/arm/junyang/golden/deploy_stack.sh 2>&1
echo "== carbon-a7ed8f95 =="
ls -lt /home/arm/junyang/carbon-a7ed8f95/ 2>&1 | head -15
echo "== files modified after 13:24 =="
find /home/arm/junyang -maxdepth 3 -newermt '2026-11-13 13:24' -not -path '*/.venv/*' -not -path '*/carbon-repo/data/bigeval_live/*' 2>/dev/null | head -20
echo "== verify_search.log size =="
ls -l /home/arm/junyang/verify_search.log
echo "DGXSTATUS1328D-END $(date '+%H:%M:%S %Z')"
