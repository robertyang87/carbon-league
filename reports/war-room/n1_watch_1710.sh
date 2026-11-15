set -x
echo "== N1WATCH-1710FAA53186 BEGIN =="
date '+%F %T %Z'
D=/root/carbon/data/rl3/snipe_n1
LG=/root/carbon/data/rl3/snipe_n1.log
W=/root/carbon/data/weights
echo "-- health pgrep --"
pgrep -af "rl3/snipe_n1" || pgrep -af snipe_n1 || echo NO_PROC
echo "-- health nvidia-smi --"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
echo "-- weights --"
ls -lt $W | head -20
md5sum $W/cnet2_m3s2r1L.npz $W/cnet2_m7bsr1L.npz $W/cnet2_m7bsg1L.npz 2>&1
echo "-- rundir snipe_n1 --"
ls -lt $D | head -40
echo "-- train.jsonl --"
cat $D/train.jsonl
echo "-- eval.jsonl --"
cat $D/eval.jsonl 2>&1 | tail -10
echo "-- EVAL lines in log --"
grep -n "EVAL" $LG | tail -20
echo "-- log mtime/size --"
ls -l $LG; date -r $LG '+%F %T %Z'
echo "-- traceback count+last --"
grep -c Traceback $LG
grep -n Traceback $LG | tail -3
echo "-- traceback exception types in new block --"
grep -n -A3 "Traceback" $LG | tail -40
echo "-- log tail 15 --"
tail -15 $LG
echo "-- config --"
cat $D/config.json
echo "-- go file --"
ls -l /tmp/n1_pool_go.txt 2>&1
echo "-- state_last --"
ls -l $D/state_last.pt 2>&1
ls -l $D/state_pre_klfix2_it14.pt 2>&1
echo "== N1WATCH-1710FAA53186 END =="
echo "-- residual ssh/scp/sftp/git procs --"
ps -eo pid,etime,args | grep -E 'ssh|scp|sftp|git' | grep -v grep || echo NONE
echo "-- last4 ckpt mtime --"
ls -td /root/carbon/data/rl3/snipe_n1/ckpt_* 2>/dev/null | head -4 | while read d; do stat -c '%y %n' "$d"; done
echo "-- snap list --"
ls -lt /root/carbon/data/rl3/snipe_n1/ | grep -i snap | head -5
