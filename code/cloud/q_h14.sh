#!/bin/bash
# host 14: finish the search-vs-pure exam (first 100 seeds per group, resumed), then car>2nd forks (d9d0d0a)
cd /root/carbon/code
until [ -f exp/fx_m2t3S20.py ]; do sleep 10; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
miss() { python3 -c "
import json,os,sys
f='../data/bigeval_live/$1.jsonl'; d={json.loads(l)['seed'] for l in open(f)} if os.path.exists(f) else set()
print(','.join(str(s) for s in range($2,$2+100) if s not in d))"; }
# --- N02 fix (REVIEW_FIXES_20261113_R2): seed-count contract + marker discipline ---
# bigeval refuses EVO_SEEDS whose count != N (exit 2 before any worker starts);
# the old dynamic calls passed N=0 and a bare `wait` still wrote SX_DONE. Now
# each dynamic list is frozen ONCE (miss runs a single time per tag, into a
# variable), N is counted from that same frozen variable and passed to
# bigeval, every PID is collected individually, the bigeval summary
# (FAIL=false, n == N) is checked where available, and SX_DONE is written only
# if ALL exams passed. Any failure (or an empty list) writes SX_FAIL and
# exits nonzero instead; the fork stage is not started.
count_seeds() { if [ -z "$1" ]; then echo 0; else awk -F, '{print NF}' <<< "$1"; fi; }
check_bigeval_health() { # $1=tag $2=want_n $3=out_file
  python3 - "$1" "$2" "$3" <<'PYEOF'
import json, os, sys
tag, want, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rec = None
p = os.path.join("..", "data", "bigeval.jsonl")
if os.path.exists(p):
    for line in open(p):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("tag") == tag:
            rec = d
if rec is not None:
    if rec.get("FAIL") is not False or rec.get("n") != want:
        print(f"HEALTH-FAIL tag={tag} FAIL={rec.get('FAIL')} n={rec.get('n')} want={want}")
        sys.exit(1)
    sys.exit(0)
try:
    txt = open(out).read() if out and os.path.exists(out) else ""
except Exception:
    txt = ""
if '"FAIL": true' in txt or '"ERROR"' in txt or "FAIL_SEARCH_ERRORS" in txt:
    print(f"HEALTH-FAIL tag={tag} (out file reports failure)")
    sys.exit(1)
print(f"HEALTH-WARN tag={tag}: no bigeval summary record found; relying on exit code only")
sys.exit(0)
PYEOF
}
collect_group() { # $1=pids-array $2=tags-array $3=wants-array $4=outs-array -> COLLECT_RC
  COLLECT_RC=0
  local _pa=$1 _ta=$2 _wa=$3 _oa=$4 _n _i _pid _tag _want _out _rc
  eval "_n=\${#$_pa[@]}"
  for ((_i = 0; _i < _n; _i++)); do
    eval "_pid=\${$_pa[$_i]}; _tag=\${$_ta[$_i]}; _want=\${$_wa[$_i]}; _out=\${$_oa[$_i]}"
    if wait "$_pid"; then _rc=0; else _rc=$?; fi
    if [ "$_rc" -ne 0 ]; then
      echo "FAIL tag=$_tag pid=$_pid exit=$_rc"
      COLLECT_RC=1
    elif ! check_bigeval_health "$_tag" "$_want" "$_out"; then
      echo "FAIL tag=$_tag (bigeval summary health check)"
      COLLECT_RC=1
    fi
  done
}
SX_PIDS=(); SX_TAGS=(); SX_WANTS=(); SX_OUTS=()
SX_RC=0
for O in cnet2_m2s2stand cnet2_m2so; do
  SEEDS=$(miss sx_S20_$O 994000)  # frozen: miss evaluated exactly once for this tag
  N=$(count_seeds "$SEEDS")
  if [ "$N" -eq 0 ]; then
    echo "SKIP sx_S20_$O: EVO_SEEDS empty (no missing games in 994000..994099 per bigeval_live) -- not counted as success"
    SX_RC=1
    continue
  fi
  echo "N02: sx_S20_$O frozen seeds=$N"
  EVO_SEEDS="$SEEDS" EVO_WORKERS=21 setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3S20.py exp/$O.py 0 "$N" sx_S20_$O > ../data/sx_S20_$O.out 2>&1 < /dev/null &
  SX_PIDS+=($!); SX_TAGS+=("sx_S20_$O"); SX_WANTS+=("$N"); SX_OUTS+=("../data/sx_S20_$O.out")
done
SEEDS=$(miss sx_S20_vs_pure 995000)  # frozen: miss evaluated exactly once for this tag
N=$(count_seeds "$SEEDS")
if [ "$N" -eq 0 ]; then
  echo "SKIP sx_S20_vs_pure: EVO_SEEDS empty (no missing games in 995000..995099 per bigeval_live) -- not counted as success"
  SX_RC=1
else
  echo "N02: sx_S20_vs_pure frozen seeds=$N"
  EVO_SEEDS="$SEEDS" AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128m2t3.npz AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}' EVO_WORKERS=21 \
    setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3S20.py awr_us_env.py 0 "$N" sx_S20_vs_pure > ../data/sx_S20_vs_pure.out 2>&1 < /dev/null &
  SX_PIDS+=($!); SX_TAGS+=("sx_S20_vs_pure"); SX_WANTS+=("$N"); SX_OUTS+=("../data/sx_S20_vs_pure.out")
fi
collect_group SX_PIDS SX_TAGS SX_WANTS SX_OUTS
if [ "$SX_RC" -ne 0 ] || [ "$COLLECT_RC" -ne 0 ]; then
  echo SX_FAIL > ../data/sx.fail
  echo "N02: search-vs-pure exam FAILED -- SX_DONE not written, fork stage not started"
  exit 1
fi
echo SX_DONE > ../data/sx.done
export AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128m2t3.npz AWR_TAU=0
export AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}' FORK_P=0.3 FORK_H=10,25,50 FORK_END=0.1 FORK_KINDS="car>2nd"
FORK_RC=0
for O in cnet2_m2s2stand cnet2_m2so s1_net; do
  FORK_TAG=car_$O ../.venv/bin/python fork_gen.py ../data/fkcar awr_us_env.py exp/$O.py 9200 100 62 > ../data/fkcar_$O.out 2>&1 || FORK_RC=1
done
if [ "$FORK_RC" -ne 0 ]; then
  echo ALL_FAIL > ../data/fkcar.fail
  echo "N02: fork stage FAILED -- ALL_DONE not written"
  exit 1
fi
echo ALL_DONE > ../data/fkcar.done
