#!/bin/bash
# C1: launch Fable g1 diagnostic chain on spot-29 (war-room approved
# insertion 23:23). Serial: g1 vs Master2r 996000, g1 vs Echo2 999000.
# Same frozen wrapper template as r1, weights swap only.
MARK="S29FG1LAUNCH2332"
echo "${MARK}_BEGIN"
date '+%F %T %Z'
echo "--- RESIDUAL SCAN ---"
ps -eo pid,ppid,etime,args | grep -E 'bigeval' | grep -v grep || echo NO-RESIDUAL-PROC
if pgrep -f 'bigeval.py exp/' > /dev/null; then
  echo "ABORT: a bigeval field is already running"
  echo "${MARK}_END"; exit 0
fi
echo "no bigeval running"
echo "--- PRE-FLIGHT MD5 ---"
EV=$(md5sum /root/carbon/code/eval_v53.py | cut -d' ' -f1)
BV=$(md5sum /root/carbon/code/bigeval.py | cut -d' ' -f1)
G1=$(md5sum /root/carbon/data/weights/cnet2_fable_g1_fromstudio.npz | cut -d' ' -f1)
TPL=$(md5sum /root/carbon/code/exp/m3s2r1.py | cut -d' ' -f1)
echo "eval_v53=$EV bigeval=$BV fable_g1=$G1 template=$TPL"
[ "$EV" = "6d24da0eeeb28bad350d8b38f08ec3fc" ] || { echo "ABORT eval_v53 md5"; echo "${MARK}_END"; exit 1; }
[ "$BV" = "a5822338d979abea807fbd2ab4f7dace" ] || { echo "ABORT bigeval md5"; echo "${MARK}_END"; exit 1; }
[ "$G1" = "f4b0e9b4894e8f43e42b0db6796c2650" ] || { echo "ABORT fable_g1 md5"; echo "${MARK}_END"; exit 1; }
[ "$TPL" = "73542cdd0ea50dce87e3638fd4c9bf5a" ] || { echo "ABORT template md5"; echo "${MARK}_END"; exit 1; }
echo "all md5 OK"
for f in fableg1_m2 fableg1_echo2; do
  if [ -e /root/carbon/data/bigeval_live/$f.jsonl ]; then
    echo "ABORT tag file already exists: $f.jsonl"; echo "${MARK}_END"; exit 1
  fi
done
echo "new tag files absent"
echo "--- WRITE WRAPPER (template, weights swap only) ---"
sed 's/cnet2_m3s2r1L\.npz/cnet2_fable_g1_fromstudio.npz/' /root/carbon/code/exp/m3s2r1.py > /root/carbon/code/exp/fableg1.py
grep -H '_W = ' /root/carbon/code/exp/fableg1.py
grep -q 'cnet2_fable_g1_fromstudio.npz' /root/carbon/code/exp/fableg1.py || { echo "ABORT swap"; echo "${MARK}_END"; exit 1; }
cd /root/carbon/code
../.venv/bin/python -m py_compile /root/carbon/code/exp/fableg1.py && echo "py_compile OK"
echo "--- WRITE CHAIN RUNNER ---"
cat > /root/carbon/data/fableg1_cal_runner.sh <<'CHAIN'
#!/bin/bash
cd /root/carbon/code
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
S=/root/carbon/data/fableg1_cal.status
echo "CHAIN_START $(date '+%F %T %Z') field1 fableg1_m2" >> $S
../.venv/bin/python bigeval.py exp/fableg1.py exp/opp_master2r.py 996000 200 fableg1_m2 >> /root/carbon/data/fableg1_m2.launcher.log 2>&1
echo "CHAIN_FIELD1_DONE rc=$? $(date '+%F %T %Z')" >> $S
echo "CHAIN_START $(date '+%F %T %Z') field2 fableg1_echo2" >> $S
../.venv/bin/python bigeval.py exp/fableg1.py exp/echo2.py 999000 200 fableg1_echo2 >> /root/carbon/data/fableg1_echo2.launcher.log 2>&1
echo "CHAIN_ALL_DONE rc=$? $(date '+%F %T %Z')" >> $S
CHAIN
chmod +x /root/carbon/data/fableg1_cal_runner.sh
rm -f /root/carbon/data/fableg1_cal.status
echo "--- LAUNCH CHAIN (detached) ---"
setsid nohup /bin/bash /root/carbon/data/fableg1_cal_runner.sh >> /root/carbon/data/fableg1_cal.out 2>&1 < /dev/null &
echo "chain launcher pid: $!"
sleep 90
echo "--- VERIFY ---"
cat /root/carbon/data/fableg1_cal.status 2>&1
echo "bigeval proc count: $(ps -eo args | grep -c '[b]igeval.py exp/')"
WPID=$(pgrep -f 'bigeval.py exp/' | tail -1)
echo "sample worker pid: $WPID"
if [ -n "$WPID" ]; then
  tr '\0' '\n' < /proc/$WPID/environ 2>/dev/null | grep -E 'EVO_WORKERS|OMP_NUM_THREADS' || echo "THREAD ENV NOT VISIBLE"
  awk '/^Threads:/{print "worker threads:", $2}' /proc/$WPID/status 2>/dev/null
fi
wc -l /root/carbon/data/bigeval_live/fableg1_m2.jsonl 2>&1
cat /proc/loadavg
echo "${MARK}_END"
