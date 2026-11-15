#!/bin/bash
MARK=S30E3LAUNCH0305
echo "$MARK PAYLOAD-BEGIN"
date '+%F %T %Z'
hostname
echo "--- SELF-SCAN ---"
if pgrep -f "bigeval.py" >/dev/null; then echo RESIDUAL-BIGEVAL-RUNNING; pgrep -af "bigeval.py"; echo "$MARK PAYLOAD-END"; exit 0; fi
echo NO-RESIDUAL
F=/root/carbon/data/packs/echo3_int8.py
echo "PACK-MD5=$(md5sum "$F" | awk '{print $1}') PACK-SIZE=$(stat -c %s "$F")"
LIVE=/root/carbon/data/bigeval_live/echo3_m2.jsonl
if [ -s "$LIVE" ]; then mv "$LIVE" "$LIVE.bak-$(date +%H%M%S)"; echo STALE-LIVE-RENAMED; fi
echo "--- LAUNCH ---"
cd /root/carbon/code || { echo NO-CODE-DIR; echo "$MARK PAYLOAD-END"; exit 0; }
EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  nohup setsid /root/carbon/.venv/bin/python bigeval.py ../data/packs/echo3_int8.py exp/opp_master2r.py 996000 200 echo3_m2 \
  > /root/carbon/data/echo3_m2.launch.log 2>&1 </dev/null &
echo "LAUNCHED-PID=$!"
sleep 25
echo "--- THREE-PIECE ---"
pgrep -af "bigeval.py" | head -5
echo -n "procs-total: "; pgrep -cf "bigeval.py" || true
wc -l "$LIVE" 2>/dev/null || echo NO-LIVE-YET
ls -l --time-style=full-iso "$LIVE" /root/carbon/data/echo3_m2.launch.log 2>/dev/null
cat /proc/loadavg
tail -3 /root/carbon/data/echo3_m2.launch.log 2>/dev/null
echo "$MARK PAYLOAD-END"
