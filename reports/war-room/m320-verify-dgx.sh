#!/bin/bash
# M320 verify-stop (runs ON DGX): confirm outer chain bash gone, state file, artifacts.
DATA=/home/arm/junyang/carbon/data
echo "VERIFY-BEGIN $(date '+%F %T %Z')"
echo '--- pid 262727 (outer chain bash) ---'
[ -d /proc/262727 ] && echo STILL-ALIVE || echo GONE
echo '--- any bash -s in carbon tree / chain leftovers ---'
for p in $(pgrep -f 'bash -s'); do echo "bash-s pid=$p cwd=$(readlink /proc/$p/cwd 2>/dev/null) etime=$(ps -o etime= -p $p)"; done
ps -eo pid,etime,args | grep -E 'cnet2_train|replica_chain|build_top|set_uw' | grep -v grep || echo NO-CHAIN-PROCS
echo '--- chain state file ---'
cat "$DATA/m9xj320_chain.state" 2>/dev/null || echo NO-STATE
echo '--- m9xj320 artifacts in data/ ---'
ls -ld "$DATA"/ds_m9xj320* "$DATA"/m9xj320_corpus* "$DATA"/M9XJ320_READY* "$DATA"/m9xj320_manifest.json 2>/dev/null
echo '--- weights (must be no m9xj320) ---'
ls "$DATA"/weights/ | grep m9xj320 || echo NO-M9XJ320-WEIGHTS
echo '--- train logs (must be no m9xj320) ---'
ls "$DATA"/dq_logs/ | grep m9xj320 || echo NO-M9XJ320-TRAINLOGS
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo "VERIFY-END"
