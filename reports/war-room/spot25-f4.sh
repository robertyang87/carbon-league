#!/bin/bash
# spot25 step F4: diagnose smoke FileNotFoundError.
MARK="SPOT25F9K3"
echo "MARKER $MARK START"
date '+%F %T %Z'
echo "--- FULL SMOKE OUT ---"
cat /root/carbon/data/spot25_smoke.out 2>/dev/null | head -60
echo "--- DATA DIR ON S25 ---"
ls -la /root/carbon/data/
echo "--- BIGEVAL PATH REFS ---"
grep -nE "open\(|os\.path|data/|\.\./" /root/carbon/code/bigeval.py | head -40
echo "MARKER $MARK END"
