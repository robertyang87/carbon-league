#!/bin/bash
# host 15 (parallel): car>2nd forks | 1-car 50-sim control | camper-bust exam vs 1C-v9 (stalled + normal seeds)
cd /root/carbon/code
for f in fx_m2t3S20n fx_m2t3S50n fx_m2t3S20cb150 fx_m2t3S20cb100; do until [ -f exp/$f.py ]; do sleep 10; done; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# --- N02 fix (REVIEW_FIXES_20261113_R2): seed-count contract + marker discipline ---
# bigeval refuses EVO_SEEDS whose count != N (exit 2 before any worker starts);
# the old camper-bust calls passed N=0, and the old `ps`-polling loops wrote
# S50_DONE/CB_DONE when processes merely disappeared, success or failure. Now
# each list is frozen ONCE in a variable, N is counted from that same variable,
# every PID is collected individually, the bigeval summary (FAIL=false,
# n == N) is checked where available, and each *_DONE marker is written only
# if its whole group passed. Any failure (or an empty list) writes the *_FAIL
# marker instead and the script exits nonzero.
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
    elif [ -n "$_want" ] && ! check_bigeval_health "$_tag" "$_want" "$_out"; then
      echo "FAIL tag=$_tag (bigeval summary health check)"
      COLLECT_RC=1
    fi
  done
}
# Frozen seed lists (single definition; counted once, never re-expanded).
SEEDS_STALLED='981000,981006,981009,981010,981011,981014,981023,981034,981039,981045,981051,981054,981055,981057,986002,986005,986007,987009,987016,987021,987037,987039,987040,987044,987051,987054,987055,987056,987057,987059,987065,987070,987089,994409,994410,994411,994412,994415,994420,994422,998410,998701,998713,998722,998727,998728,998734,998738,998741,998742,998755,998759,998792,998820,998829,998877,998899'
SEEDS_NORMAL='987000,987001,987002,987003,987004,987005,987006,987007,987008,987010,987011,987012,987013,987014,987015,987017,987018,987019,987020,987022,987023,987024,987025,987026,987027,987028,987029,987030,987031,987032'
N_STALLED=$(count_seeds "$SEEDS_STALLED")
N_NORMAL=$(count_seeds "$SEEDS_NORMAL")
echo "N02: frozen seed lists stalled=$N_STALLED normal=$N_NORMAL (expect 57/30)"
( export AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128m2t3.npz AWR_TAU=0
  export AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}' FORK_P=0.3 FORK_H=10,25,50 FORK_END=0.1 FORK_KINDS="car>2nd"
  _frc=0
  for O in cnet2_m2s2stand cnet2_m2so s1_net; do
    FORK_TAG=car_$O ../.venv/bin/python fork_gen.py ../data/fkcar awr_us_env.py exp/$O.py 9200 100 20 > ../data/fkcar_$O.out 2>&1 || _frc=1
  done; exit $_frc ) &
FORK_PIDS=($!); FORK_TAGS=("fork_fkcar"); FORK_WANTS=(); FORK_OUTS=("")
S50_PIDS=(); S50_TAGS=(); S50_WANTS=(); S50_OUTS=()
for O in cnet2_m2s2stand cnet2_m2so s1_net; do
  EVO_WORKERS=7 setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3S50n.py exp/$O.py 994000 50 sx_S50_$O > ../data/sx_S50_$O.out 2>&1 < /dev/null &
  S50_PIDS+=($!); S50_TAGS+=("sx_S50_$O"); S50_WANTS+=(50); S50_OUTS+=("../data/sx_S50_$O.out")
done
CB_PIDS=(); CB_TAGS=(); CB_WANTS=(); CB_OUTS=()
CB_RC=0
if [ "$N_STALLED" -eq 0 ]; then echo "SKIP cbst_*: seed list SEEDS_STALLED is empty -- not counted as success"; CB_RC=1; fi
if [ "$N_NORMAL" -eq 0 ]; then echo "SKIP cbno_*: seed list SEEDS_NORMAL is empty -- not counted as success"; CB_RC=1; fi
for A in S20n S20cb150 S20cb100; do
  if [ "$N_STALLED" -gt 0 ]; then
    EVO_SEEDS="$SEEDS_STALLED" EVO_WORKERS=5 setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3$A.py exp/opp_1cz_v9.py 0 "$N_STALLED" cbst_$A > ../data/cbst_$A.out 2>&1 < /dev/null &
    CB_PIDS+=($!); CB_TAGS+=("cbst_$A"); CB_WANTS+=("$N_STALLED"); CB_OUTS+=("../data/cbst_$A.out")
  fi
  if [ "$N_NORMAL" -gt 0 ]; then
    EVO_SEEDS="$SEEDS_NORMAL" EVO_WORKERS=3 setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3$A.py exp/opp_1cz_v9.py 0 "$N_NORMAL" cbno_$A > ../data/cbno_$A.out 2>&1 < /dev/null &
    CB_PIDS+=($!); CB_TAGS+=("cbno_$A"); CB_WANTS+=("$N_NORMAL"); CB_OUTS+=("../data/cbno_$A.out")
  fi
done
OVERALL_RC=0
collect_group FORK_PIDS FORK_TAGS FORK_WANTS FORK_OUTS
if [ "$COLLECT_RC" -ne 0 ]; then
  echo ALL_FAIL > ../data/fkcar.fail
  echo "N02: fork stage FAILED -- ALL_DONE not written"
  OVERALL_RC=1
else
  echo ALL_DONE > ../data/fkcar.done
fi
collect_group S50_PIDS S50_TAGS S50_WANTS S50_OUTS
if [ "$COLLECT_RC" -ne 0 ]; then
  echo S50_FAIL > ../data/s50.fail
  echo "N02: S50 exam FAILED -- S50_DONE not written"
  OVERALL_RC=1
else
  echo S50_DONE > ../data/s50.done
fi
collect_group CB_PIDS CB_TAGS CB_WANTS CB_OUTS
if [ "$CB_RC" -ne 0 ] || [ "$COLLECT_RC" -ne 0 ]; then
  echo CB_FAIL > ../data/cb.fail
  echo "N02: camper-bust queue FAILED -- CB_DONE not written"
  OVERALL_RC=1
else
  echo CB_DONE > ../data/cb.done
fi
exit $OVERALL_RC
