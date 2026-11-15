#!/bin/bash
echo "MARK WARROOMSYNC2151_STUDIO"
date '+%F %T %Z'
ps aux | grep -E 'eval_v53|bigeval|pilot|q_du' | grep -v grep | head -8
echo "--- q progress files ---"
ls -la ~/carbon/*q_du* ~/carbon/*pilot_du* 2>/dev/null | head -10
find /Users/arm -maxdepth 4 -name '*q_du_formal*' -o -maxdepth 4 -name '*pilot_du_formal*' 2>/dev/null | head -10
echo "--- disk ---"
df -h / 2>/dev/null | tail -1
