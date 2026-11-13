#!/bin/bash
# replica_chain.sh -- SINGLE-CHAIN replica pipeline (2026-11-12, anti-idle rule).
#
# One script, one executor, start to finish:
#   wait READY -> count assertion -> build -> set_uw -> dual-arm launch -> attestation
# A replica job must NEVER be split into "dataset agent finishes, training waits
# for a second dispatch" again (M7Bs v2 lesson: ds_m7bs_v2 built 17:31, GPU idle
# 16:51->17:58, executor silent past deadline; see reports/live/replica_gap_plan.md).
#
# COMPLETION DEFINITION (binding): the executor is DONE only when the training
# trio is LIVE (both PIDs registered + GPU busy + fresh logs/weights) AND the
# attestation block (processes/GPU/logs/counts) is printed into this script's
# own log. "Dataset built" is NOT completion. The receipt must be delivered
# within one watch cycle (15 min) of launch -- M7Bs v2 also proved the reverse
# failure: training WAS launched 17:58 but the receipt never arrived, which is
# indistinguishable from not launching. Silence = handoff not established.
#
# Usage (run from code/, on the training box):
#   PREFIX=m7bsv2 TEACHER='RyanX-AIPowercore-M7Bs-b010' \
#   CORPUS=$DATA/m7bs_corpus MATCHES=$DATA/m7bs_matches.json DS=$DATA/ds_m7bs_v2 \
#   EXPECTED=142 ./replica_chain.sh 2>&1 | tee $DATA/dq_logs/replica_chain_m7bsv2.log
#
# Required env: PREFIX, TEACHER (exact leaderboard name), CORPUS, DS
# Optional env: MATCHES (matches .json for the count assertion),
#   EXPECTED (expected games; default = index entries after assertion),
#   READY_FILE (wait until this file exists), WAIT_MIN (default 60),
#   SYNC_CMD (API re-pull command run ONCE inside this same run when counts
#             mismatch, e.g. "../.venv/bin/python ladder_corpus.py $CORPUS '<teacher-id>'";
#             ladder_corpus.sync() re-pulls matches+replays from the ladder API
#             and rewrites index.json),
#   REPO_DIR + REPO_MATCHES (repo copy of the matches list; on mismatch the
#             repo is pulled (timeout 60) and the fresher matches list copied in
#             before the API re-pull attempt),
#   EPOCHS (default 20), BC_THREADS (default 5), BC_DEVICE (default cuda).
set -u
D=${DATA:-/home/arm/junyang/carbon/data}
: "${PREFIX:?set PREFIX e.g. m7bsv2}"
: "${TEACHER:?set TEACHER exact leaderboard name}"
: "${CORPUS:?set CORPUS dir (contains index.json and replays/)}"
: "${DS:?set DS output dataset dir}"
EPOCHS=${EPOCHS:-20}; WAIT_MIN=${WAIT_MIN:-60}
LOGD=$D/dq_logs; mkdir -p "$LOGD" "$D/weights"
mark() { echo "REPLICA-CHAIN[$PREFIX] $* $(date '+%F %T %Z')"; }

counts() {  # prints "replays matches index"
  local rp=$(ls "$CORPUS"/replays/*.json.gz 2>/dev/null | wc -l)
  local mt=-1; [ -n "${MATCHES:-}" ] && [ -f "$MATCHES" ] && \
    mt=$(../.venv/bin/python -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$MATCHES" 2>/dev/null || echo -1)
  local ix=$(../.venv/bin/python -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$CORPUS/index.json" 2>/dev/null || echo -1)
  echo "$rp $mt $ix"
}

echo '--- 1) WAIT-READY (file + GPU free; network ops timeout-wrapped) ---'
for i in $(seq 1 "$WAIT_MIN"); do
  READY=1
  [ -n "${READY_FILE:-}" ] && [ ! -f "$READY_FILE" ] && READY=0
  pgrep -f 'cnet2_train[.]py' >/dev/null 2>&1 && READY=0
  [ "$READY" = 1 ] && { mark "READY at wait-min $i"; break; }
  [ "$i" = "$WAIT_MIN" ] && { mark "FAIL not READY after ${WAIT_MIN}min"; exit 1; }
  sleep 60
done

echo '--- 2) COUNT ASSERTION: replays == matches == index (no orphans downstream) ---'
read RP MT IX <<< "$(counts)"; echo "counts: replays=$RP matches=$MT index=$IX expected=${EXPECTED:-auto}"
if [ "$RP" != "$IX" ] || { [ -n "${MATCHES:-}" ] && [ "$MT" != "$IX" ]; }; then
  echo "MISMATCH -> same-run repair: repo pull + API re-pull of matches, then recount"
  if [ -n "${REPO_DIR:-}" ]; then timeout 60 git -C "$REPO_DIR" pull --ff-only 2>&1 | tail -2 || true; fi
  if [ -n "${REPO_MATCHES:-}" ] && [ -n "${MATCHES:-}" ]; then cp "$REPO_MATCHES" "$MATCHES" || true; fi
  if [ -n "${SYNC_CMD:-}" ]; then eval "$SYNC_CMD" || true; fi
  read RP MT IX <<< "$(counts)"; echo "counts after repair: replays=$RP matches=$MT index=$IX"
fi
if [ "$RP" != "$IX" ] || { [ -n "${MATCHES:-}" ] && [ "$MT" != "$IX" ]; } || [ "$IX" -le 0 ]; then
  mark "FAIL counts still misaligned (replays=$RP matches=$MT index=$IX) -- build NOT started, no orphans passed on"; exit 1
fi
if [ -n "${EXPECTED:-}" ] && [ "$IX" != "$EXPECTED" ]; then
  mark "FAIL index=$IX != EXPECTED=$EXPECTED -- build NOT started"; exit 1
fi
mark "COUNTS-OK replays=$RP matches=$MT index=$IX"

echo '--- 3) BUILD + SET_UW ---'
rm -rf "$DS"
CNET2_TOP_TEACHERS="${TEACHER}@2" ../.venv/bin/python build_top.py "$DS" "$CORPUS" > "$LOGD/build_${PREFIX}.log" 2>&1 \
  || { BRC=$?; grep '^C02-FAIL' "$LOGD/build_${PREFIX}.log" 2>/dev/null || true; mark "FAIL build_top rc=$BRC (see $LOGD/build_${PREFIX}.log)"; exit 1; }
# C02 declaration gate (BEFORE set_uw): the builder's meta.json declares
# failed_games (original game id + reason) and valid_games (games that
# contributed >= 1 row). A declared partial dataset is refused here --
# set_uw must never bless it as a training input -- and the count assertion
# is on VALID games vs the index. A legitimate zero-row game is not a
# failure, but it contributes no rows, so it does not count as valid
# either; zero-row ids are reported separately so the failure is diagnosable.
GATE=$(../.venv/bin/python -c "
import json,sys
d=json.load(open(sys.argv[1]))
f=d.get('failed_games')
if f is None:
    print('NOFIELD 0 - -')
else:
    zero=','.join(g.get('key','?') for g in d.get('games',[]) if g.get('n',0)==0 and not g.get('failed')) or '-'
    print('%s %d %s %s' % (d.get('valid_games',-1), len(f), ','.join(x.get('key','?') for x in f) or '-', zero))
" "$DS/meta.json" 2>/dev/null)
read GAMES NFAILED FAILED_IDS ZERO_IDS <<< "$GATE"
echo "dataset valid_games=$GAMES failed_games=$NFAILED failed_ids=$FAILED_IDS zero_row_ids=$ZERO_IDS index=$IX"
[ -n "${GAMES:-}" ] && [ "$GAMES" != "NOFIELD" ] \
  || { mark "FAIL meta.json missing/unreadable or predates the C02 contract (no failed_games declaration) -- set_uw NOT run"; exit 1; }
[ "$NFAILED" = 0 ] \
  || { mark "FAIL dataset declares $NFAILED failed game(s): $FAILED_IDS -- partial dataset refused, set_uw NOT run"; exit 1; }
[ "$GAMES" = "$IX" ] || { mark "FAIL dataset valid games=$GAMES != index=$IX (zero-row games: ${ZERO_IDS:-none})"; exit 1; }
../.venv/bin/python set_uw.py "$DS" 1 1 1 >> "$LOGD/build_${PREFIX}.log" 2>&1 \
  || { mark "FAIL set_uw rc=$?"; exit 1; }
# C02: set_uw only reads the merged npy files, so a non-zero rc is not the
# only failure mode -- confirm it actually wrote the weight files.
[ -f "$DS/w.npy" ] && [ -f "$DS/uw.npy" ] \
  || { mark "FAIL set_uw produced no w.npy/uw.npy in $DS"; exit 1; }

echo '--- 4) DUAL-ARM LAUNCH (standard recipe, 20 epochs) ---'
export BC_DEVICE=${BC_DEVICE:-cuda} BC_RAM=1 BC_THREADS=${BC_THREADS:-5} BC_USE_W=1 BC_USE_UW=1 \
       BC_BS=256 BC_SAVE_EPOCHS=1 BC_PCT=0.1 BC_LR=3e-4
unset BC_AMP BC_CLIP BC_TRAIN_PREFIX BC_CACHE_TRUNK BC_SELF_KL BC_EXIT_W BC_AUX_W BC_COMPILE BC_CAR_KL
BC_INIT=$D/weights/cnet2_arch_n1_last.npz nohup ../.venv/bin/python cnet2_train.py "$DS" "$D/weights/cnet2_${PREFIX}r1.npz" "$EPOCHS" \
  '{"C":128,"NB":8,"D":128,"ZI":1,"TA":4,"TNH":4}' > "$LOGD/${PREFIX}r1.log" 2>&1 &
PID_R1=$!
BC_INIT=$D/weights/cnet2_b128m2t3.npz nohup ../.venv/bin/python cnet2_train.py "$DS" "$D/weights/cnet2_${PREFIX}g1.npz" "$EPOCHS" \
  '{"C":128,"NB":8,"D":128}' > "$LOGD/${PREFIX}g1.log" 2>&1 &
PID_G1=$!
sleep 5; echo "pids: r1=$PID_R1 g1=$PID_G1 (all train procs: $(pgrep -f 'cnet2_train[.]py' | tr '\n' ' '))"

echo '--- 5) ATTESTATION (completion = trio live + this block in the log) ---'
# R2 (Codex re-review 20261114): freshness snapshot. Per-arm liveness
# (P2-4) still passes an arm that printed ep0 and then hung -- alive,
# but its log and weights never move again. So before the proof-window
# sleep, snapshot each arm's log mtime and newest-weight mtime;
# LAUNCH-OK below requires BOTH to have advanced during the window,
# per arm. Platform-agnostic on purpose: mtimes only (stat in GNU and
# BSD form), no GPU probe -- this chain also runs on boxes without
# NVIDIA tooling, so GPU health is attested in the printout above for
# a human, never as a pass/fail leg.
mt_of() { stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo 0; }
newest_wt() { ls -t "$D/weights/cnet2_${PREFIX}$1"*.npz 2>/dev/null | head -1; }
WT_R1=$(newest_wt r1); WT_G1=$(newest_wt g1)
SNAP_LOG_R1=$(mt_of "$LOGD/${PREFIX}r1.log"); SNAP_LOG_G1=$(mt_of "$LOGD/${PREFIX}g1.log")
SNAP_WT_R1=0; [ -n "$WT_R1" ] && SNAP_WT_R1=$(mt_of "$WT_R1")
SNAP_WT_G1=0; [ -n "$WT_G1" ] && SNAP_WT_G1=$(mt_of "$WT_G1")
echo "freshness snapshot (pre-window): r1 log=$SNAP_LOG_R1 wt=$SNAP_WT_R1; g1 log=$SNAP_LOG_G1 wt=$SNAP_WT_G1"
sleep 120
EP0R=$(grep -c '"ep": 0' "$LOGD/${PREFIX}r1.log" 2>/dev/null || true)
EP0G=$(grep -c '"ep": 0' "$LOGD/${PREFIX}g1.log" 2>/dev/null || true)
echo "=== ATTESTATION $PREFIX $(date '+%F %T %Z') ==="
echo "processes:"; ps -eo pid,etime,args | grep 'cnet2_train' | grep -v grep || echo NO-TRAIN-PROC
echo "gpu:"; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null || echo NO-NVIDIA-SMI
echo "logs:"; ls -l --time-style=+%F_%T "$LOGD/${PREFIX}r1.log" "$LOGD/${PREFIX}g1.log" 2>/dev/null
echo "weights:"; ls -lt --time-style=+%F_%T "$D/weights/cnet2_${PREFIX}"*.npz 2>/dev/null | head -6
read RP MT IX <<< "$(counts)"; echo "counts: replays=$RP matches=$MT index=$IX dataset_games=$GAMES"
echo "ep0 lines: r1=$EP0R g1=$EP0G"
# P2-4 (Codex review 20261114): liveness is PER ARM, judged on the PID each
# arm was actually launched with (PID_R1 / PID_G1 above). The previous third
# leg -- "any cnet2_train process alive anywhere" -- let a dead arm pass
# whenever the other arm (or an unrelated train process) was still up: with
# g1 dead after ep0 the chain still printed LAUNCH-OK.
ALIVE_R1=0; [ -n "${PID_R1:-}" ] && kill -0 "$PID_R1" 2>/dev/null && ALIVE_R1=1
ALIVE_G1=0; [ -n "${PID_G1:-}" ] && kill -0 "$PID_G1" 2>/dev/null && ALIVE_G1=1
echo "arm liveness: r1 pid=${PID_R1:-?} alive=$ALIVE_R1; g1 pid=${PID_G1:-?} alive=$ALIVE_G1"
# R2 freshness verdict: each arm's log AND newest weight must carry a
# strictly newer mtime than the pre-window snapshot -- an arm alive
# but frozen (no log line, no weight save during the whole window)
# fails here and is named, with the stuck artifact values printed.
WT_R1=$(newest_wt r1); WT_G1=$(newest_wt g1)
NOW_LOG_R1=$(mt_of "$LOGD/${PREFIX}r1.log"); NOW_LOG_G1=$(mt_of "$LOGD/${PREFIX}g1.log")
NOW_WT_R1=0; [ -n "$WT_R1" ] && NOW_WT_R1=$(mt_of "$WT_R1")
NOW_WT_G1=0; [ -n "$WT_G1" ] && NOW_WT_G1=$(mt_of "$WT_G1")
FRESH_R1=1; { [ "$NOW_LOG_R1" -gt "$SNAP_LOG_R1" ] && [ "$NOW_WT_R1" -gt "$SNAP_WT_R1" ]; } || FRESH_R1=0
FRESH_G1=1; { [ "$NOW_LOG_G1" -gt "$SNAP_LOG_G1" ] && [ "$NOW_WT_G1" -gt "$SNAP_WT_G1" ]; } || FRESH_G1=0
echo "arm freshness: r1 log $SNAP_LOG_R1->$NOW_LOG_R1 wt $SNAP_WT_R1->$NOW_WT_R1 fresh=$FRESH_R1; g1 log $SNAP_LOG_G1->$NOW_LOG_G1 wt $SNAP_WT_G1->$NOW_WT_G1 fresh=$FRESH_G1"
if [ "${EP0R:-0}" -lt 1 ] || [ "${EP0G:-0}" -lt 1 ] || [ "$ALIVE_R1" != 1 ] || [ "$ALIVE_G1" != 1 ] || [ "$FRESH_R1" != 1 ] || [ "$FRESH_G1" != 1 ]; then
  DEAD_ARMS=""; [ "$ALIVE_R1" != 1 ] && DEAD_ARMS="$DEAD_ARMS r1(pid=${PID_R1:-?})"
  [ "$ALIVE_G1" != 1 ] && DEAD_ARMS="$DEAD_ARMS g1(pid=${PID_G1:-?})"
  STALE_ARMS=""; [ "$FRESH_R1" != 1 ] && STALE_ARMS="$STALE_ARMS r1(log:$SNAP_LOG_R1->$NOW_LOG_R1 wt:$SNAP_WT_R1->$NOW_WT_R1)"
  [ "$FRESH_G1" != 1 ] && STALE_ARMS="$STALE_ARMS g1(log:$SNAP_LOG_G1->$NOW_LOG_G1 wt:$SNAP_WT_G1->$NOW_WT_G1)"
  mark "LAUNCH-FAIL attestation: dead arm(s):${DEAD_ARMS:- none}; stale arm(s):${STALE_ARMS:- none} (ep0 r1=$EP0R g1=$EP0G) -- NOT complete, escalate with this log"; exit 1
fi
mark "LAUNCH-OK + ATTESTED -- executor complete ONLY in this sense; training itself continues to ep$((EPOCHS-1))"
