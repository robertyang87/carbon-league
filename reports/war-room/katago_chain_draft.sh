#!/bin/bash
# katago_chain_draft.sh -- SINGLE-CHAIN KataGo-generation pipeline (DRAFT, 2026-11-13).
#
# Modeled on code/replica_chain.sh (same five-step skeleton, same completion
# definition); the ONLY structural difference is step 0/1: the corpus comes
# from self-play generation (selfplay_gen.py shards, foundation ring 1)
# instead of a ladder-replay pull. Everything downstream of the corpus --
# count assertion, build, set_uw, dual-arm launch, attestation -- is the
# replica chain verbatim in spirit, because the anti-recurrence rules are
# chain-agnostic (2026-11-12 owner rulings, AGENTS.md):
#   (1) build-dataset + launch-training in ONE script chain, never split
#       into "dataset done, training waits for a second dispatch";
#   (2) three-way count assertion replays == matches == index, mismatch
#       repaired in the SAME run (repo pull + API re-pull of matches);
#   (3) completion = training trio LIVE (both PIDs + GPU busy + fresh
#       logs/weights) AND the attestation block printed into this script's
#       own log. "Dataset built" is NOT completion.
#
# Five steps:
#   0) (optional) GENERATE: if GEN_WEIGHTS is set, run selfplay_gen.py to
#      produce/refresh the corpus shard in $CORPUS before anything else.
#   1) WAIT-READY: READY_FILE present + no cnet2_train running (+ GPU free
#      is implied by no trainer); network ops timeout-wrapped.
#   2) COUNT ASSERTION over all corpus shards (+ optional matches list).
#   3) BUILD (fail-closed: build_top.py in N03 semantics -- an indexed
#      replay that is missing FAILS the build; this script deliberately
#      does NOT set CNET2_ALLOW_PARTIAL) + assert meta.games == index.
#   4) SET_UW: set_uw.py $DS 1 1 1.
#   5) DUAL-ARM LAUNCH (r1 = attention arm, g1 = control arm, standard BC
#      recipe from foundation_taskbook ring 5) -> ATTESTATION.
#
# ============================ FILL ON-MACHINE ============================
# Required:
#   PREFIX    dataset/arm name prefix, e.g. gen1 (weights land as
#             cnet2_gen1r1.npz / cnet2_gen1g1.npz, dataset ds_$PREFIX or $DS)
#   TEACHER   exact generating-policy name; used as CNET2_TOP_TEACHERS=
#             "${TEACHER}@2". MUST equal the name the self-play corpus was
#             generated with (selfplay_gen --name / GEN_NAME), character-
#             exact, or build keeps zero sides.
#   CORPUS    primary corpus shard dir (index.json + replays/)
#   DS        output dataset dir
# Optional data/repair:
#   EXTRA_SHARDS  space-separated additional corpus shard dirs (merged
#             into counts and the build_top invocation)
#   MATCHES   matches .json for the count assertion (for self-play shards
#             this is usually unset: replays == index is the binding pair,
#             and EXPECTED pins the total)
#   EXPECTED  expected total games (index entries across shards)
#   DATA      data root (default /home/arm/junyang/carbon/data)
#   SYNC_CMD  API re-pull command run ONCE in-run on count mismatch
#   REPO_DIR + REPO_MATCHES  repo checkout + its matches copy, pulled
#             (timeout 60) and copied in before the API re-pull attempt
# Optional readiness:
#   READY_FILE, WAIT_MIN (default 60)
# Optional generation (step 0; all unset = skip):
#   GEN_WEIGHTS   bare-net npz for selfplay_gen.py
#   GEN_SEED0 / GEN_SEED1   seed range (disjoint per shard/worker!)
#   GEN_NAME      side name written into the corpus (default $TEACHER)
#   GEN_WORKERS   default min(nproc-2, 56)
#   GEN_MAX_GAMES / GEN_MAX_SECONDS  optional caps (unset = whole seed
#             range; never pass 0 -- selfplay_gen treats it as stop-at-0)
#   GEN_SCRIPT    path to selfplay_gen.py (default ./selfplay_gen.py --
#             the probe scripts drop it into code/; foundation ring 1
#            定版 md5 b275e952da14d644e6fab41e4ce941c8)
# Arms:
#   INIT_R1   attention-arm BC_INIT npz (default $DATA/weights/cnet2_arch_n1_last.npz)
#   INIT_G1   control-arm  BC_INIT npz (default $DATA/weights/cnet2_b128m2t3.npz)
#   CFG_R1    default {"C":128,"NB":8,"D":128,"ZI":1,"TA":4,"TNH":4}
#   CFG_G1    default {"C":128,"NB":8,"D":128}
#   EPOCHS (default 20), BC_DEVICE (default cuda; Studio: mps),
#   BC_THREADS (default 5)
# ========================================================================
#
# Usage (run from code/, on the training box):
#   PREFIX=gen1 TEACHER='m3s2r1' CORPUS=$DATA/gen1_shard0 \
#   DS=$DATA/ds_gen1 EXPECTED=2000 ./katago_chain_draft.sh 2>&1 | tee $DATA/dq_logs/katago_chain_gen1.log
#
# DRAFT STATUS: written locally 2026-11-13 against the local style copy of
# replica_chain.sh (hidden_files copy md5 51081dae...; main tip is
# 07fa5e91... -- diff against main's copy once on-machine before first use).
# Never run anywhere yet; bash -n clean only.
set -u
D=${DATA:-/home/arm/junyang/carbon/data}
: "${PREFIX:?set PREFIX e.g. gen1}"
: "${TEACHER:?set TEACHER exact generating-policy name}"
: "${CORPUS:?set CORPUS shard dir (contains index.json and replays/)}"
: "${DS:?set DS output dataset dir}"
EPOCHS=${EPOCHS:-20}; WAIT_MIN=${WAIT_MIN:-60}
LOGD=$D/dq_logs; mkdir -p "$LOGD" "$D/weights"
mark() { echo "KATAGO-CHAIN[$PREFIX] $* $(date '+%F %T %Z')"; }
SHARDS="$CORPUS ${EXTRA_SHARDS:-}"

counts() {  # prints "replays matches index" summed over all shards
  local rp=0 ix=0 d n
  for d in $SHARDS; do
    n=$(ls "$d"/replays/*.json.gz 2>/dev/null | wc -l); rp=$((rp+n))
    n=$(../.venv/bin/python -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$d/index.json" 2>/dev/null || echo 0); ix=$((ix+n))
  done
  local mt=-1; [ -n "${MATCHES:-}" ] && [ -f "$MATCHES" ] && \
    mt=$(../.venv/bin/python -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$MATCHES" 2>/dev/null || echo -1)
  echo "$rp $mt $ix"
}

if [ -n "${GEN_WEIGHTS:-}" ]; then
  echo '--- 0) GENERATE (optional self-play shard refresh, ring 1) ---'
  : "${GEN_SEED0:?set GEN_SEED0 when GEN_WEIGHTS is set}"
  : "${GEN_SEED1:?set GEN_SEED1 when GEN_WEIGHTS is set}"
  GEN_WORKERS=${GEN_WORKERS:-$(( $(nproc) - 2 ))}; [ "$GEN_WORKERS" -gt 56 ] && GEN_WORKERS=56
  # NOTE: selfplay_gen stops at games >= --max-games, so 0 would generate
  # NOTHING -- only pass the caps when explicitly set; seed range bounds it.
  GEN_CAPS=""; [ -n "${GEN_MAX_GAMES:-}" ] && GEN_CAPS="$GEN_CAPS --max-games $GEN_MAX_GAMES"
  [ -n "${GEN_MAX_SECONDS:-}" ] && GEN_CAPS="$GEN_CAPS --max-seconds $GEN_MAX_SECONDS"
  ../.venv/bin/python "${GEN_SCRIPT:-./selfplay_gen.py}" --weights "$GEN_WEIGHTS" \
    --seed0 "$GEN_SEED0" --seed1 "$GEN_SEED1" --out "$CORPUS" --workers "$GEN_WORKERS" \
    $GEN_CAPS \
    --name "${GEN_NAME:-$TEACHER}" > "$LOGD/gen_${PREFIX}.log" 2>&1 \
    || { mark "FAIL selfplay_gen rc=$? (see $LOGD/gen_${PREFIX}.log)"; exit 1; }
  grep -E "SPGEN-RESULT" "$LOGD/gen_${PREFIX}.log" | tail -1 || true
  mark "GENERATE done"
fi

echo '--- 1) WAIT-READY (file + no trainer running; network ops timeout-wrapped) ---'
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

echo '--- 3) BUILD (fail-closed; CNET2_ALLOW_PARTIAL deliberately NOT set) ---'
rm -rf "$DS"
CNET2_TOP_TEACHERS="${TEACHER}@2" ../.venv/bin/python build_top.py "$DS" $SHARDS > "$LOGD/build_${PREFIX}.log" 2>&1 \
  || { mark "FAIL build_top rc=$? (see $LOGD/build_${PREFIX}.log)"; exit 1; }
GAMES=$(../.venv/bin/python -c "import json,sys;d=json.load(open(sys.argv[1]));g=d.get('games');print(len(g) if isinstance(g,list) else g)" "$DS/meta.json" 2>/dev/null)
echo "dataset games=$GAMES"
[ "$GAMES" = "$IX" ] || { mark "FAIL dataset games=$GAMES != index=$IX"; exit 1; }

echo '--- 4) SET_UW ---'
../.venv/bin/python set_uw.py "$DS" 1 1 1 >> "$LOGD/build_${PREFIX}.log" 2>&1 \
  || { mark "FAIL set_uw rc=$?"; exit 1; }
mark "BUILD+SET_UW OK games=$GAMES"

echo '--- 5) DUAL-ARM LAUNCH (standard BC recipe, ring 5) + ATTESTATION ---'
export BC_DEVICE=${BC_DEVICE:-cuda} BC_RAM=1 BC_THREADS=${BC_THREADS:-5} BC_USE_W=1 BC_USE_UW=1 \
       BC_BS=256 BC_SAVE_EPOCHS=1 BC_PCT=0.1 BC_LR=3e-4
unset BC_AMP BC_CLIP BC_TRAIN_PREFIX BC_CACHE_TRUNK BC_SELF_KL BC_EXIT_W BC_AUX_W BC_COMPILE BC_CAR_KL
BC_INIT=${INIT_R1:-$D/weights/cnet2_arch_n1_last.npz} nohup ../.venv/bin/python cnet2_train.py "$DS" "$D/weights/cnet2_${PREFIX}r1.npz" "$EPOCHS" \
  "${CFG_R1:-{\"C\":128,\"NB\":8,\"D\":128,\"ZI\":1,\"TA\":4,\"TNH\":4}}" > "$LOGD/${PREFIX}r1.log" 2>&1 &
BC_INIT=${INIT_G1:-$D/weights/cnet2_b128m2t3.npz} nohup ../.venv/bin/python cnet2_train.py "$DS" "$D/weights/cnet2_${PREFIX}g1.npz" "$EPOCHS" \
  "${CFG_G1:-{\"C\":128,\"NB\":8,\"D\":128}}" > "$LOGD/${PREFIX}g1.log" 2>&1 &
sleep 5; echo "pids: $(pgrep -f 'cnet2_train[.]py' | tr '\n' ' ')"

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
if [ "${EP0R:-0}" -lt 1 ] || [ "${EP0G:-0}" -lt 1 ] || ! pgrep -f 'cnet2_train[.]py' >/dev/null 2>&1; then
  mark "FAIL attestation: trio not live (ep0 r1=$EP0R g1=$EP0G) -- NOT complete, escalate with this log"; exit 1
fi
mark "LAUNCH-OK + ATTESTED -- executor complete ONLY in this sense; training itself continues to ep$((EPOCHS-1))"
