#!/bin/bash
MARK=N1WATCH-1740B7E2F91A
echo "MARK $MARK START $(date '+%F %T %Z')"
D=/root/carbon/data/rl3/snipe_n1
LOG=/root/carbon/data/rl3/snipe_n1.log
echo "=== weights ==="
ls -lt /root/carbon/data/weights/ | head -20
md5sum /root/carbon/data/weights/cnet2_m3s2r1L.npz /root/carbon/data/weights/cnet2_m7bsr1L.npz /root/carbon/data/weights/cnet2_m7bsg1L.npz 2>&1
echo "=== proc ==="
pgrep -af "rl3/snipe_n1" || echo "NO-PROC"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== dir ==="
ls -lt "$D" | head -25
echo "=== ckpt mtimes (last 6) ==="
ls -ldt "$D"/ckpt_* 2>/dev/null | head -6
stat -c '%y %n' "$D"/ckpt_last "$D"/state_last "$D"/train.jsonl "$D"/league.json 2>/dev/null
echo "=== train.jsonl all rows (it pre_s upd_s winrate kl dlogp kl_coef t amp) ==="
python3 - "$D/train.jsonl" <<'PY'
import json,sys
p=sys.argv[1]
try:
    rows=[json.loads(l) for l in open(p) if l.strip()]
except Exception as e:
    print("TRAIN-READ-ERR", e); sys.exit(0)
for r in rows:
    print(r.get("it"), r.get("pre_s"), r.get("upd_s"), r.get("winrate"), r.get("kl"), r.get("dlogp"), r.get("kl_coef"), r.get("t"), r.get("amp"))
PY
echo "=== eval.jsonl ==="
ls -l "$D"/eval.jsonl 2>/dev/null; cat "$D"/eval.jsonl 2>/dev/null | tail -5
echo "=== EVAL lines in log ==="
grep -a "EVAL" "$LOG" 2>/dev/null | tail -15 || echo "NO-EVAL-LINES"
echo "=== traceback ==="
grep -ac "Traceback" "$LOG" 2>/dev/null; stat -c '%y %s' "$LOG" 2>/dev/null
tail -3 "$LOG" 2>/dev/null
echo "=== snaps ==="
ls -ldt "$D"/snap_* 2>/dev/null | head -5
echo "=== config pool/amp ==="
python3 - "$D/config.json" <<'PY'
import json,sys
try:
    c=json.load(open(sys.argv[1]))
    print("amp=", c.get("amp"), "kl=", c.get("kl"), "pool=", c.get("pool") or c.get("opponents") or c.get("league"))
except Exception as e:
    print("CONFIG-READ-ERR", e)
PY
echo "=== pool go flag ==="
ls -l /tmp/n1_pool_go.txt 2>&1
echo "=== residual ssh/scp/sftp/git procs ==="
ps -eo pid,etime,args | grep -E "ssh|scp|sftp|git" | grep -v grep | grep -v "rl3/snipe_n1" || echo "NONE-EXCLUDING-TRAINING"
echo "MARK $MARK END $(date '+%F %T %Z')"
